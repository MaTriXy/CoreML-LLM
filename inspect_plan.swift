#!/usr/bin/env swift
// MLComputePlan inspector: shows per-op device assignment AND cost estimates
// Usage: swift inspect_plan.swift <model.mlmodelc> [--cpu-ane] [--verbose]

import Foundation
import CoreML

guard CommandLine.arguments.count > 1 else {
    print("Usage: inspect_plan.swift <path/to/model.mlmodelc> [--cpu-ane] [--verbose]")
    exit(1)
}

func deviceName(_ d: MLComputeDevice) -> String {
    switch d {
    case .cpu: return "CPU"
    case .gpu: return "GPU"
    case .neuralEngine: return "ANE"
    @unknown default: return "?"
    }
}

let url = URL(fileURLWithPath: CommandLine.arguments[1])
let cpuAne = CommandLine.arguments.contains("--cpu-ane")
let verbose = CommandLine.arguments.contains("--verbose")

let sem = DispatchSemaphore(value: 0)
var exitCode: Int32 = 0

Task {
    defer { sem.signal() }
    do {
        let config = MLModelConfiguration()
        config.computeUnits = cpuAne ? .cpuAndNeuralEngine : .all
        print("Loading: \(url.lastPathComponent), mode: \(cpuAne ? "CPU+NE" : "ALL")")
        let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)

        guard case .program(let program) = plan.modelStructure else {
            print("Not a program")
            return
        }

        for (fnName, fn) in program.functions {
            print("\n===== Function: \(fnName) =====")
            let block = fn.block

            var deviceCounts: [String: Int] = [:]
            var opTypeDevice: [String: [String: Int]] = [:]
            // Track when ANE was supported but GPU chosen
            var missedAne: [(String, Double, Double)] = []  // (opType, aneCost, gpuCost)

            for op in block.operations {
                let t = op.operatorName
                if t.contains("const") || t.contains("lut_to_dense") { continue }

                let usage = plan.deviceUsage(for: op)
                guard let u = usage else { continue }
                let pref = deviceName(u.preferred)
                deviceCounts[pref, default: 0] += 1
                opTypeDevice[t, default: [:]][pref, default: 0] += 1

                let supports = u.supported
                let supportsAne = supports.contains { if case .neuralEngine = $0 { return true }; return false }

                if supportsAne && pref != "ANE" {
                    // ANE possible but not chosen — investigate cost
                    let cost = plan.estimatedCost(of: op)
                    let w = cost?.weight ?? 0
                    missedAne.append((t, w, w))  // we only get overall cost, not per-device
                }
            }

            print("\n--- Device summary ---")
            for (k, v) in deviceCounts.sorted(by: { $0.value > $1.value }) {
                print("  \(k): \(v) ops")
            }

            print("\n--- Per op-type device breakdown ---")
            for (opType, devices) in opTypeDevice.sorted(by: { $0.key < $1.key }) {
                let parts = devices.map { "\($0.key):\($0.value)" }.joined(separator: " ")
                print("  \(opType): \(parts)")
            }

            // Show top costly ops that COULD have run on ANE but didn't
            if !missedAne.isEmpty {
                print("\n--- Top ANE-capable ops not chosen for ANE (by cost) ---")
                let top = missedAne.sorted { $0.1 > $1.1 }.prefix(15)
                for (t, c, _) in top {
                    print("  \(t): cost=\(String(format: "%.4f", c))")
                }
                let totalCost = missedAne.reduce(0.0) { $0 + $1.1 }
                print("  TOTAL ANE-missed cost: \(String(format: "%.4f", totalCost))")
            }

            if verbose {
                print("\n--- All ops ---")
                for op in block.operations {
                    let t = op.operatorName
                    if t.contains("const") || t.contains("lut_to_dense") { continue }
                    let usage = plan.deviceUsage(for: op)
                    guard let u = usage else { continue }
                    let pref = deviceName(u.preferred)
                    let sup = u.supported.map(deviceName).joined(separator: ",")
                    let cost = plan.estimatedCost(of: op)?.weight ?? 0
                    let out = op.outputs.first?.name ?? "?"
                    print("  \(t) [\(out)]: pref=\(pref) sup=[\(sup)] cost=\(String(format: "%.4f", cost))")
                }
            }
        }
    } catch {
        print("Error: \(error)")
        exitCode = 1
    }
}

sem.wait()
exit(exitCode)
