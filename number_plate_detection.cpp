#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <chrono>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

namespace fs = std::filesystem;

enum class LogLevel {
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

class Logger {
public:
    static void init(const std::string& logFilePath) {
        getInstance().logFileStream.open(logFilePath, std::ios::app);
    }

    static void log(LogLevel level, const std::string& sender, const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");

        std::string levelStr;
        switch (level) {
            case LogLevel::INFO:     levelStr = "INFO"; break;
            case LogLevel::WARNING:  levelStr = "WARNING"; break;
            case LogLevel::ERROR:    levelStr = "ERROR"; break;
            case LogLevel::CRITICAL: levelStr = "CRITICAL"; break;
        }

        std::string formatted = ss.str() + " [" + levelStr + "] " + sender + ": " + message;
        std::cout << formatted << std::endl;

        auto& inst = getInstance();
        if (inst.logFileStream.is_open()) {
            inst.logFileStream << formatted << std::endl;
        }
    }

private:
    Logger() = default;
    ~Logger() {
        if (logFileStream.is_open()) {
            logFileStream.close();
        }
    }
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }
    std::ofstream logFileStream;
};

class PipelineException : public std::runtime_error {
public:
    explicit PipelineException(const std::string& message) : std::runtime_error(message) {}
};

class InferenceException : public PipelineException {
public:
    explicit InferenceException(const std::string& message) : PipelineException(message) {}
};

class ProcessingException : public PipelineException {
public:
    explicit ProcessingException(const std::string& message) : PipelineException(message) {}
};

struct PipelineConfig {
    std::string onnxModelPath = "runs/detect/train/weights/best.onnx";
    std::string inputImagesDir = "License-Plate-Data/try";
    std::string cropsDir = "runs/crops";
    std::string warpedDir = "runs/warped";
    std::string badPolygonsDir = "runs/bad_polygons";
    std::string goodPolygonsDir = "runs/good_polygons";
    std::string cannyDir = "runs/canny";
    std::string tessDataPath = "/usr/share/tesseract-ocr/4.00/tessdata";
    std::string ocrLanguage = "eng";
    std::string platesCsvPath = "runs/plates.csv";
    std::string comparisonCsvPath = "runs/comparison_result.csv";
    float confThreshold = 0.45f;
    float nmsThreshold = 0.50f;
    int networkInputWidth = 640;
    int networkInputHeight = 640;
};

class PlateDetector {
public:
    explicit PlateDetector(std::string modelPath, float confThresh, float nmsThresh, int netWidth, int netHeight)
        : modelPath_(std::move(modelPath)),
          confThreshold_(confThresh),
          nmsThreshold_(nmsThresh),
          inputWidth_(netWidth),
          inputHeight_(netHeight) {}

    void loadModel() {
        try {
            if (!fs::exists(modelPath_)) {
                throw InferenceException("ONNX model file not found: " + modelPath_);
            }
            net_ = cv::dnn::readNetFromONNX(modelPath_);
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            Logger::log(LogLevel::INFO, "PlateDetector", "YOLO ONNX model successfully loaded: " + modelPath_);
        } catch (const cv::Exception& e) {
            Logger::log(LogLevel::ERROR, "PlateDetector", "OpenCV DNN error: " + std::string(e.what()));
            throw InferenceException(std::string(e.what()));
        }
    }

    std::vector<fs::path> detectAndCrop(const fs::path& sourceDir, const fs::path& outputDir) {
        fs::create_directories(outputDir);
        std::vector<fs::path> savedCrops;

        if (!fs::exists(sourceDir)) {
            Logger::log(LogLevel::ERROR, "PlateDetector", "Source directory does not exist: " + sourceDir.string());
            return savedCrops;
        }

        for (const auto& entry : fs::directory_iterator(sourceDir)) {
            if (!entry.is_regular_file()) continue;

            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

            cv::Mat image = cv::imread(entry.path().string());
            if (image.empty()) {
                Logger::log(LogLevel::WARNING, "PlateDetector", "Unable to decode: " + entry.path().string());
                continue;
            }

            try {
                std::vector<cv::Rect> bboxes = runInference(image);
                for (size_t i = 0; i < bboxes.size(); ++i) {
                    cv::Rect box = bboxes[i] & cv::Rect(0, 0, image.cols, image.rows);
                    if (box.width <= 0 || box.height <= 0) continue;

                    cv::Mat cropped = image(box);
                    std::string outName;
                    if (bboxes.size() > 1) {
                        outName = entry.path().stem().string() + "_" + std::to_string(i) + entry.path().extension().string();
                    } else {
                        outName = entry.path().filename().string();
                    }

                    fs::path outPath = outputDir / outName;
                    if (cv::imwrite(outPath.string(), cropped)) {
                        savedCrops.push_back(outPath);
                    } else {
                        Logger::log(LogLevel::WARNING, "PlateDetector", "Failed to write crop: " + outPath.string());
                    }
                }
            } catch (const std::exception& e) {
                Logger::log(LogLevel::ERROR, "PlateDetector", "Inference failure on " + entry.path().string() + ": " + e.what());
            }
        }

        Logger::log(LogLevel::INFO, "PlateDetector", "Completed cropping. Total crops saved: " + std::to_string(savedCrops.size()));
        return savedCrops;
    }

private:
    std::vector<cv::Rect> runInference(const cv::Mat& image) {
        cv::Mat blob;
        cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(inputWidth_, inputHeight_), cv::Scalar(), true, false);
        net_.setInput(blob);

        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());

        cv::Mat output = outputs[0];
        if (output.dims == 3) {
            output = output.reshape(1, output.size[1]);
            cv::transpose(output, output);
        }

        int rows = output.rows;
        int dimensions = output.cols;

        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        float xFactor = static_cast<float>(image.cols) / inputWidth_;
        float yFactor = static_cast<float>(image.rows) / inputHeight_;

        float* data = reinterpret_cast<float*>(output.data);
        for (int i = 0; i < rows; ++i) {
            float* row = data + i * dimensions;
            float* scores = row + 4;
            cv::Mat scoresMat(1, dimensions - 4, CV_32FC1, scores);
            cv::Point classIdPoint;
            double maxClassScore;
            cv::minMaxLoc(scoresMat, nullptr, &maxClassScore, nullptr, &classIdPoint);

            if (maxClassScore >= confThreshold_) {
                float cx = row[0];
                float cy = row[1];
                float w = row[2];
                float h = row[3];

                int left = static_cast<int>((cx - 0.5f * w) * xFactor);
                int top = static_cast<int>((cy - 0.5f * h) * yFactor);
                int width = static_cast<int>(w * xFactor);
                int height = static_cast<int>(h * yFactor);

                boxes.emplace_back(left, top, width, height);
                confidences.push_back(static_cast<float>(maxClassScore));
                classIds.push_back(classIdPoint.x);
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confThreshold_, nmsThreshold_, indices);

        std::vector<cv::Rect> resultBoxes;
        for (int idx : indices) {
            resultBoxes.push_back(boxes[idx]);
        }
        return resultBoxes;
    }

    std::string modelPath_;
    float confThreshold_;
    float nmsThreshold_;
    int inputWidth_;
    int inputHeight_;
    cv::dnn::Net net_;
};

class ImageRectifier {
public:
    ImageRectifier(fs::path cannyDir, fs::path warpedDir, fs::path goodPolyDir, fs::path badPolyDir)
        : cannyDir_(std::move(cannyDir)),
          warpedDir_(std::move(warpedDir)),
          goodPolyDir_(std::move(goodPolyDir)),
          badPolyDir_(std::move(badPolyDir)) {
        fs::create_directories(cannyDir_);
        fs::create_directories(warpedDir_);
        fs::create_directories(goodPolyDir_);
        fs::create_directories(badPolyDir_);
    }

    static std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point2f>& pts) {
        if (pts.size() != 4) {
            throw ProcessingException("Requires exactly 4 corner points to sort.");
        }

        std::vector<cv::Point2f> sortedX = pts;
        std::sort(sortedX.begin(), sortedX.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.x < b.x;
        });

        std::vector<cv::Point2f> left = {sortedX[0], sortedX[1]};
        std::vector<cv::Point2f> right = {sortedX[2], sortedX[3]};

        std::sort(left.begin(), left.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.y < b.y;
        });
        cv::Point2f tl = left[0];
        cv::Point2f bl = left[1];

        std::sort(right.begin(), right.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.y < b.y;
        });
        cv::Point2f tr = right[0];
        cv::Point2f br = right[1];

        return {tl, tr, br, bl};
    }

    static std::vector<cv::Point> mergeCloseEdges(const std::vector<cv::Point>& poly, float angleThresh = 15.0f, float distThresh = 10.0f) {
        size_t n = poly.size();
        if (n < 3) return poly;

        struct Edge {
            cv::Point pt1;
            cv::Point pt2;
            float angle;
        };

        std::vector<Edge> edges;
        for (size_t i = 0; i < n; ++i) {
            cv::Point p1 = poly[i];
            cv::Point p2 = poly[(i + 1) % n];
            float angle = std::atan2(static_cast<float>(p2.y - p1.y), static_cast<float>(p2.x - p1.x)) * 180.0f / CV_PI;
            edges.push_back({p1, p2, angle});
        }

        std::vector<Edge> merged;
        for (const auto& edge : edges) {
            bool matched = false;
            for (auto& m : merged) {
                float d1 = cv::norm(edge.pt1 - m.pt1);
                float d2 = cv::norm(edge.pt1 - m.pt2);
                if (std::abs(edge.angle - m.angle) < angleThresh && (d1 < distThresh || d2 < distThresh)) {
                    m.pt1 = (m.pt1 + edge.pt1) * 0.5;
                    m.pt2 = (m.pt2 + edge.pt2) * 0.5;
                    m.angle = (m.angle + edge.angle) * 0.5f;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                merged.push_back(edge);
            }
        }

        std::vector<cv::Point> result;
        for (const auto& e : merged) {
            result.push_back(e.pt1);
        }
        return result;
    }

    cv::Mat rectify(const cv::Mat& src, const std::vector<cv::Point2f>& corners) {
        std::vector<cv::Point2f> ordered = orderPoints(corners);
        cv::Point2f tl = ordered[0];
        cv::Point2f tr = ordered[1];
        cv::Point2f br = ordered[2];
        cv::Point2f bl = ordered[3];

        double widthA = cv::norm(br - bl);
        double widthB = cv::norm(tr - tl);
        double longSide = (widthA + widthB) / 2.0;

        double heightA = cv::norm(tr - br);
        double heightB = cv::norm(tl - bl);
        double shortSide = (heightA + heightB) / 2.0;

        if (shortSide <= 0.0) {
            throw ProcessingException("Degenerate quadrilateral with non-positive dimension.");
        }

        double sideRatio = longSide / shortSide;
        double ratioLimit = 1.75;
        int outputWidth = (sideRatio > ratioLimit) ? 253 : 130;
        int outputHeight = (sideRatio > ratioLimit) ? 50 : 95;

        std::vector<cv::Point2f> dst = {
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(static_cast<float>(outputWidth - 1), 0.0f),
            cv::Point2f(static_cast<float>(outputWidth - 1), static_cast<float>(outputHeight - 1)),
            cv::Point2f(0.0f, static_cast<float>(outputHeight - 1))
        };

        cv::Mat M = cv::getPerspectiveTransform(ordered, dst);
        cv::Mat warped;
        cv::warpPerspective(src, warped, M, cv::Size(outputWidth, outputHeight));
        return warped;
    }

    void processCropBatch(const std::vector<fs::path>& cropPaths) {
        for (size_t idx = 0; idx < cropPaths.size(); ++idx) {
            const auto& path = cropPaths[idx];
            cv::Mat img = cv::imread(path.string());
            if (img.empty()) {
                Logger::log(LogLevel::WARNING, "ImageRectifier", "Unable to read: " + path.string());
                continue;
            }

            cv::Mat gray, blurred, edges;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
            cv::Canny(blurred, edges, 50, 150);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(edges, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

            bool goodPolyFound = false;
            std::vector<cv::Point> largestPoly;

            for (int step = 0; step < 15 && !goodPolyFound; ++step) {
                std::vector<std::pair<double, std::vector<cv::Point>>> candidates;
                for (const auto& cnt : contours) {
                    double delta = 0.01 + step * 0.005;
                    double epsilon = delta * cv::arcLength(cnt, true);
                    std::vector<cv::Point> approx;
                    cv::approxPolyDP(cnt, approx, epsilon, true);
                    std::vector<cv::Point> merged = mergeCloseEdges(approx);

                    if (merged.size() == 4 && cv::isContourConvex(merged)) {
                        double area = cv::contourArea(merged);
                        candidates.emplace_back(area, merged);
                    }
                }

                if (!candidates.empty()) {
                    std::sort(candidates.begin(), candidates.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });

                    const auto& bestPoly = candidates[0].second;
                    cv::Rect bound = cv::boundingRect(bestPoly);

                    if (bound.width > 0.6 * img.cols && bound.height > 0.5 * img.rows) {
                        goodPolyFound = true;
                        largestPoly = bestPoly;
                    }
                }
            }

            int goodFlag = goodPolyFound ? 1 : 0;
            cv::imwrite((cannyDir_ / ("image_" + std::to_string(goodFlag) + "_" + std::to_string(idx) + ".jpg")).string(), edges);

            if (goodPolyFound) {
                cv::Mat polyImage = img.clone();
                std::vector<std::vector<cv::Point>> polys = {largestPoly};
                cv::drawContours(polyImage, polys, 0, cv::Scalar(0, 255, 0), 4);
                cv::imwrite((goodPolyDir_ / ("image_" + std::to_string(goodFlag) + "_" + std::to_string(idx) + ".jpg")).string(), polyImage);

                std::vector<cv::Point2f> corners;
                for (const auto& pt : largestPoly) {
                    corners.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
                }

                try {
                    cv::Mat warped = rectify(img, corners);
                    cv::imwrite((warpedDir_ / path.filename()).string(), warped);
                } catch (const std::exception& e) {
                    Logger::log(LogLevel::ERROR, "ImageRectifier", "Rectification failed on " + path.string() + ": " + e.what());
                    cv::imwrite((warpedDir_ / path.filename()).string(), img);
                }
            } else {
                cv::imwrite((badPolyDir_ / ("image_" + std::to_string(goodFlag) + "_" + std::to_string(idx) + ".jpg")).string(), img);
                cv::imwrite((warpedDir_ / path.filename()).string(), img);
            }

            if ((idx + 1) % 50 == 0 || idx == cropPaths.size() - 1) {
                Logger::log(LogLevel::INFO, "ImageRectifier", "Rectified " + std::to_string(idx + 1) + " / " + std::to_string(cropPaths.size()));
            }
        }
    }

private:
    fs::path cannyDir_;
    fs::path warpedDir_;
    fs::path goodPolyDir_;
    fs::path badPolyDir_;
};

class PlateOCRProcessor {
public:
    PlateOCRProcessor(const std::string& tessDataPath, const std::string& language) {
        tessApi_ = std::make_unique<tesseract::TessBaseAPI>();
        if (tessApi_->Init(tessDataPath.c_str(), language.c_str())) {
            Logger::log(LogLevel::CRITICAL, "PlateOCRProcessor", "Could not initialize Tesseract with lang: " + language);
            throw PipelineException("Tesseract initialization failure");
        }
        tessApi_->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
        tessApi_->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-");
        Logger::log(LogLevel::INFO, "PlateOCRProcessor", "Tesseract OCR engine successfully initialized.");
    }

    ~PlateOCRProcessor() {
        if (tessApi_) {
            tessApi_->End();
        }
    }

    static std::string fixCommonMistakes(const std::string& raw) {
        std::string fixed = raw;
        for (size_t i = 0; i < fixed.size(); ++i) {
            char c = fixed[i];
            if (i < 3) {
                if (c == '0') fixed[i] = 'O';
                else if (c == '1') fixed[i] = 'I';
                else if (c == '5') fixed[i] = 'S';
            } else {
                if (c == 'O' || c == 'D') fixed[i] = '0';
                else if (c == 'I') fixed[i] = '1';
                else if (c == 'S') fixed[i] = '5';
            }
        }
        return fixed;
    }

    static std::string cleanPlateText(const std::string& raw) {
        std::string cleaned;
        for (char c : raw) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                cleaned += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            }
        }

        if (cleaned.length() < 6) {
            return "";
        }

        std::string plateCandidate = cleaned.substr(cleaned.length() - 6);
        plateCandidate = fixCommonMistakes(plateCandidate);
        return plateCandidate.substr(0, 3) + "-" + plateCandidate.substr(3);
    }

    std::string extractText(const cv::Mat& image) {
        if (image.empty()) return "";

        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }

        cv::Mat processed;
        cv::resize(gray, processed, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);
        cv::threshold(processed, processed, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        tessApi_->SetImage(processed.data, processed.cols, processed.rows, 1, static_cast<int>(processed.step));
        char* outText = tessApi_->GetUTF8Text();
        std::string result = outText ? std::string(outText) : "";
        delete[] outText;

        return cleanPlateText(result);
    }

    void processDirectoryToCsv(const fs::path& inputDir, const fs::path& outputCsv) {
        std::map<std::string, std::vector<std::string>> results;

        for (const auto& entry : fs::directory_iterator(inputDir)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

            cv::Mat img = cv::imread(entry.path().string());
            if (img.empty()) continue;

            std::string plate = extractText(img);
            if (!plate.empty()) {
                std::string filename = entry.path().stem().string();
                size_t pos = filename.find('_');
                std::string coreName = (pos != std::string::npos) ? filename.substr(0, pos) : filename;
                results[coreName].push_back(plate);
            } else {
                Logger::log(LogLevel::WARNING, "PlateOCRProcessor", "Unrecognized or invalid plate syntax: " + entry.path().filename().string());
            }
        }

        fs::create_directories(outputCsv.parent_path());
        std::ofstream csvFile(outputCsv.string());
        if (!csvFile.is_open()) {
            Logger::log(LogLevel::ERROR, "PlateOCRProcessor", "Failed to write CSV at " + outputCsv.string());
            throw PipelineException("Unable to open output CSV file");
        }

        csvFile << "plate1;plate2;filename\n";
        for (const auto& [name, plates] : results) {
            std::string p1 = plates[0];
            std::string p2 = (plates.size() > 1) ? plates[1] : "";
            csvFile << p1 << ";" << p2 << ";" << name << "\n";
        }
        csvFile.close();
        Logger::log(LogLevel::INFO, "PlateOCRProcessor", "Exported plates to " + outputCsv.string());
    }

private:
    std::unique_ptr<tesseract::TessBaseAPI> tessApi_;
};

class PipelineEvaluator {
public:
    static double evaluate(const fs::path& groundTruthCsv, const fs::path& generatedCsv, const fs::path& outputCsv) {
        std::map<std::string, std::string> detected;

        std::ifstream genFile(generatedCsv.string());
        if (!genFile.is_open()) {
            Logger::log(LogLevel::ERROR, "PipelineEvaluator", "Cannot open generated CSV: " + generatedCsv.string());
            return 0.0;
        }

        std::string line;
        std::getline(genFile, line);
        while (std::getline(genFile, line)) {
            std::stringstream ss(line);
            std::string p1, p2, fileKey;
            if (std::getline(ss, p1, ';') && std::getline(ss, p2, ';') && std::getline(ss, fileKey, ';')) {
                detected[fileKey] = p1;
            }
        }
        genFile.close();

        std::ifstream gtFile(groundTruthCsv.string());
        if (!gtFile.is_open()) {
            Logger::log(LogLevel::ERROR, "PipelineEvaluator", "Cannot open Ground Truth CSV: " + groundTruthCsv.string());
            return 0.0;
        }

        fs::create_directories(outputCsv.parent_path());
        std::ofstream outComp(outputCsv.string());
        outComp << "Original;OCR;Filename;Match\n";

        int matches = 0;
        int total = 0;

        while (std::getline(gtFile, line)) {
            std::stringstream ss(line);
            std::string origPlate, dummy1, dummy2, url;
            if (std::getline(ss, origPlate, ';') &&
                std::getline(ss, dummy1, ';') &&
                std::getline(ss, dummy2, ';') &&
                std::getline(ss, url, ';')) {

                size_t oPos = url.find("/o/");
                if (oPos != std::string::npos) {
                    std::string segment = url.substr(oPos + 3);
                    size_t dotPos = segment.find(".jpg");
                    std::string fileKey = (dotPos != std::string::npos) ? segment.substr(0, dotPos) : segment;

                    total++;
                    std::string detectedPlate = detected.count(fileKey) ? detected[fileKey] : "";
                    bool isMatch = (!detectedPlate.empty() && detectedPlate == origPlate);

                    if (isMatch) matches++;
                    outComp << origPlate << ";" << detectedPlate << ";" << fileKey << ";"
                            << (isMatch ? "Match" : "No Match") << "\n";
                }
            }
        }
        gtFile.close();
        outComp.close();

        double accuracy = (total > 0) ? (static_cast<double>(matches) / total * 100.0) : 0.0;
        Logger::log(LogLevel::INFO, "PipelineEvaluator", "Evaluation accuracy: " + std::to_string(accuracy) + "% (" +
                    std::to_string(matches) + "/" + std::to_string(total) + ")");
        return accuracy;
    }
};

class LicensePlatePipeline {
public:
    explicit LicensePlatePipeline(PipelineConfig config)
        : config_(std::move(config)),
          detector_(config_.onnxModelPath, config_.confThreshold, config_.nmsThreshold, config_.networkInputWidth, config_.networkInputHeight),
          rectifier_(config_.cannyDir, config_.warpedDir, config_.goodPolygonsDir, config_.badPolygonsDir),
          ocrProcessor_(config_.tessDataPath, config_.ocrLanguage) {}

    void initialize() {
        detector_.loadModel();
    }

    void execute() {
        try {
            Logger::log(LogLevel::INFO, "LicensePlatePipeline", "Step 1: Detecting and cropping plates");
            std::vector<fs::path> crops = detector_.detectAndCrop(config_.inputImagesDir, config_.cropsDir);

            Logger::log(LogLevel::INFO, "LicensePlatePipeline", "Step 2: Rectifying perspectives");
            rectifier_.processCropBatch(crops);

            Logger::log(LogLevel::INFO, "LicensePlatePipeline", "Step 3: Extracting characters via OCR");
            ocrProcessor_.processDirectoryToCsv(config_.warpedDir, config_.platesCsvPath);

            Logger::log(LogLevel::INFO, "LicensePlatePipeline", "Pipeline execution finished successfully.");
        } catch (const PipelineException& e) {
            Logger::log(LogLevel::CRITICAL, "LicensePlatePipeline", "Pipeline halted due to error: " + std::string(e.what()));
            throw;
        }
    }

    void evaluate(const std::string& groundTruthCsv) {
        PipelineEvaluator::evaluate(groundTruthCsv, config_.platesCsvPath, config_.comparisonCsvPath);
    }

private:
    PipelineConfig config_;
    PlateDetector detector_;
    ImageRectifier rectifier_;
    PlateOCRProcessor ocrProcessor_;
};

int main(int argc, char** argv) {
    Logger::init("pipeline_execution.log");
    Logger::log(LogLevel::INFO, "Main", "Starting ANPR application execution");

    try {
        PipelineConfig config;
        LicensePlatePipeline pipeline(config);
        pipeline.initialize();
        pipeline.execute();

        if (argc > 1) {
            std::string groundTruthCsv = argv[1];
            pipeline.evaluate(groundTruthCsv);
        }
    } catch (const std::exception& e) {
        Logger::log(LogLevel::CRITICAL, "Main", "Fatal failure encountered: " + std::string(e.what()));
        return EXIT_FAILURE;
    }

    Logger::log(LogLevel::INFO, "Main", "Application terminated normally");
    return EXIT_SUCCESS;
}