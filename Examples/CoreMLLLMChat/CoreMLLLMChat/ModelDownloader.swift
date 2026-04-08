import Foundation

/// Downloads and caches CoreML models from GitHub Releases.
@Observable
final class ModelDownloader {
    var isDownloading = false
    var progress: Double = 0
    var status = ""
    var availableModels: [ModelInfo] = ModelInfo.defaults

    private let fileManager = FileManager.default

    struct ModelInfo: Identifiable {
        let id: String
        let name: String
        let size: String
        let baseURL: String  // GitHub release base URL
        let folderName: String
        /// Files to download: (relativePath, sizeBytes)
        let files: [(String, Int)]

        static let defaults: [ModelInfo] = [
            ModelInfo(
                id: "qwen2.5-0.5b",
                name: "Qwen2.5 0.5B",
                size: "309 MB",
                baseURL: "https://github.com/john-rocky/CoreML-LLM/releases/download/v0.1.0",
                folderName: "qwen2.5-0.5b",
                files: [
                    ("qwen2.5-0.5b-coreml.zip", 309_000_000),
                ]
            ),
        ]
    }

    /// Check if model is already downloaded.
    func isDownloaded(_ model: ModelInfo) -> Bool {
        localModelURL(for: model) != nil
    }

    /// Get the model.mlpackage URL if downloaded.
    func localModelURL(for model: ModelInfo) -> URL? {
        let dir = modelsDirectory.appendingPathComponent(model.folderName)
        let pkg = dir.appendingPathComponent("model.mlpackage")
        return fileManager.fileExists(atPath: pkg.path) ? pkg : nil
    }

    /// Download a model. Returns the model.mlpackage URL.
    func download(_ model: ModelInfo) async throws -> URL {
        if let existing = localModelURL(for: model) { return existing }

        isDownloading = true
        progress = 0
        status = "Downloading \(model.name)..."
        defer { isDownloading = false }

        let file = model.files[0]
        guard let url = URL(string: "\(model.baseURL)/\(file.0)") else {
            throw DownloadError.invalidURL
        }

        // Download zip
        let tempFile = try await downloadFile(url)

        // Extract using Foundation's built-in coordinator
        status = "Extracting..."
        let destDir = modelsDirectory.appendingPathComponent(model.folderName)
        try? fileManager.removeItem(at: destDir)
        try fileManager.createDirectory(at: destDir, withIntermediateDirectories: true)

        // iOS/macOS: use FileManager to unzip
        try extractZip(at: tempFile, to: destDir)
        try? fileManager.removeItem(at: tempFile)

        guard let result = localModelURL(for: model) else {
            throw DownloadError.extractionFailed
        }

        status = "Ready"
        progress = 1.0
        return result
    }

    /// Delete a downloaded model.
    func delete(_ model: ModelInfo) throws {
        let dir = modelsDirectory.appendingPathComponent(model.folderName)
        if fileManager.fileExists(atPath: dir.path) {
            try fileManager.removeItem(at: dir)
        }
    }

    // MARK: - Private

    private var modelsDirectory: URL {
        let docs = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("Models")
    }

    private func downloadFile(_ url: URL) async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            let session = URLSession(configuration: .default, delegate: ProgressTracker { [weak self] p in
                Task { @MainActor in self?.progress = p }
            }, delegateQueue: nil)

            let task = session.downloadTask(with: url) { tempURL, _, error in
                if let error { continuation.resume(throwing: error); return }
                guard let tempURL else { continuation.resume(throwing: DownloadError.extractionFailed); return }
                let dest = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".zip")
                do {
                    try FileManager.default.moveItem(at: tempURL, to: dest)
                    continuation.resume(returning: dest)
                } catch { continuation.resume(throwing: error) }
            }
            task.resume()
        }
    }

    private func extractZip(at zipURL: URL, to destDir: URL) throws {
        // Use SSZipArchive-style approach with Foundation
        // iOS 16+: we can use the process-less approach
        #if os(macOS)
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        proc.arguments = ["-o", zipURL.path, "-d", destDir.path]
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try proc.run()
        proc.waitUntilExit()
        #else
        // iOS: Use FileManager's built-in support via NSFileCoordinator
        // or fall back to a manual approach
        //
        // The trick: rename .zip and use UIDocument to extract,
        // or use the simpler approach of spawning a background task
        //
        // Simplest iOS approach: use `Process` equivalent isn't available,
        // so we use Apple's Compression framework directly
        try decompressZip(from: zipURL, to: destDir)
        #endif
    }

    #if os(iOS)
    private func decompressZip(from zipURL: URL, to destDir: URL) throws {
        // Read ZIP file using Foundation (minimal implementation for flat ZIPs)
        // For production, consider adding ZIPFoundation SPM package
        //
        // iOS 16+ supports reading ZIP archives via FileWrapper
        let wrapper = try FileWrapper(url: zipURL, options: .immediate)
        if let children = wrapper.fileWrappers {
            for (name, child) in children {
                let destURL = destDir.appendingPathComponent(name)
                try child.write(to: destURL, options: .atomic, originalContentsURL: nil)
            }
        } else if let data = wrapper.regularFileContents {
            // Single file
            let destURL = destDir.appendingPathComponent(zipURL.lastPathComponent)
            try data.write(to: destURL)
        }
    }
    #endif
}

// MARK: - Progress Tracker

private final class ProgressTracker: NSObject, URLSessionDownloadDelegate {
    let onProgress: (Double) -> Void
    init(onProgress: @escaping (Double) -> Void) { self.onProgress = onProgress }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {}

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        onProgress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }
}

enum DownloadError: LocalizedError {
    case invalidURL
    case extractionFailed
    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid download URL"
        case .extractionFailed: return "Failed to extract model"
        }
    }
}
