document.addEventListener("DOMContentLoaded", function () {
  // =========================================================
  // MAIN ELEMENTS
  // =========================================================

  const form = document.getElementById("quantizeForm");
  const fileInput = document.getElementById("modelFile");
  const fileLabel = document.getElementById("fileLabel");
  const modelSource = document.getElementById("modelSource");
  const uploadModelSource = document.getElementById("uploadModelSource");
  const huggingFaceModelSource = document.getElementById("huggingFaceModelSource");
  const hfModelId = document.getElementById("hfModelId");
  const modelDropzone = document.getElementById("modelDropzone");
  const datasetDropzone = document.getElementById("datasetDropzone");
  const quantizeButton = document.getElementById("quantizeButton");
  const workflowMetric = document.getElementById("workflowMetric");
  const dashboardPages = document.getElementById("dashboardPages");
  const dashboardTabs = document.querySelectorAll("[data-dashboard-target]");
  const resultsIndicator = document.getElementById("resultsIndicator");
  const output = document.getElementById("output");

  const quantization = document.getElementById("quantization");
  const deviceMap = document.getElementById("deviceMap");

  const statusValue = document.getElementById("statusValue");
  const deviceValue = document.getElementById("deviceValue");
  const memoryValue = document.getElementById("memoryValue");

  const datasetInput = document.getElementById("datasetFile");
  const datasetLabel = document.getElementById("datasetLabel");
  const datasetHelp = document.getElementById("datasetHelp");
  let selectedDatasetFile = null;

  if (datasetInput) {
    datasetInput.addEventListener("change", function (e) {
      const file = e.target.files && e.target.files[0];
      if (file) {
        setDatasetFile(file);
      } else {
        setDatasetFile(null);
      }
    });
  }

  console.log("Quantization Studio main.js loaded");

  // =========================================================
  // GLOBAL STATE
  // =========================================================

  let currentDeviceInfo = null;
  let selectedModelFile = null;
  let uploadedModelFile = null;
  let lastQuantizationResult = null;

  function setWorkflowState(state, label) {
    if (workflowMetric) workflowMetric.dataset.state = state;
    if (statusValue) statusValue.textContent = label;
  }

  function setDashboardPage(pageName) {
    dashboardTabs.forEach((tab) => {
      const isActive = tab.dataset.dashboardTarget === pageName;
      tab.classList.toggle("is-active", isActive);
      tab.setAttribute("aria-selected", String(isActive));
      tab.tabIndex = isActive ? 0 : -1;
    });

    document.querySelectorAll("[data-dashboard-page]").forEach((page) => {
      const isActive = page.dataset.dashboardPage === pageName;
      page.classList.toggle("is-active", isActive);
      page.hidden = !isActive;
    });
  }

  dashboardTabs.forEach((tab, index) => {
    tab.addEventListener("click", () => {
      setDashboardPage(tab.dataset.dashboardTarget);
    });
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
      event.preventDefault();
      const direction = event.key === "ArrowRight" ? 1 : -1;
      const nextTab = dashboardTabs[(index + direction + dashboardTabs.length) % dashboardTabs.length];
      nextTab.focus();
      setDashboardPage(nextTab.dataset.dashboardTarget);
    });
  });

  function setModelFile(file) {
    uploadedModelFile = file || null;
    selectedModelFile = file || null;
    if (fileLabel) {
      fileLabel.textContent = file
        ? file.name
        : "Drop a model or ZIP bundle here";
    }
    modelDropzone?.classList.toggle("has-file", Boolean(file));
    if (file) {
      setWorkflowState("ready", "Ready");
      output.className = "output";
      output.innerHTML = `
        <div class="output-mark">OK</div>
        <h3>Model selected</h3>
        <p>${escapeHtml(file.name)}</p>
        <p>${formatFileSize(file.size)}</p>
      `;
    }
    updateDeviceRecommendation();
  }

  function setDatasetFile(file) {
    selectedDatasetFile = file || null;
    if (datasetLabel) {
      datasetLabel.textContent = file
        ? file.name
        : "Built-in SST-2 Benchmark (Default)";
    }
    if (datasetHelp) {
      datasetHelp.textContent = file
        ? `Custom dataset loaded (${(file.size / 1024).toFixed(1)} KB)`
        : "or click to upload a custom .json or .csv evaluation dataset";
    }
    datasetDropzone?.classList.toggle("has-file", Boolean(file));
  }

  function enableFileDrop(dropzone, onFile) {
    if (!dropzone) return;
    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("is-dragging");
      });
    });
    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("is-dragging");
      });
    });
    dropzone.addEventListener("drop", (event) => {
      const file = event.dataTransfer?.files?.[0];
      if (file) onFile(file);
    });
  }

  enableFileDrop(modelDropzone, (file) => {
    setModelFile(file);
  });
  enableFileDrop(datasetDropzone, (file) => {
    setDatasetFile(file);
  });

  function updateModelSource() {
    const isHuggingFace = modelSource?.value === "huggingface";
    if (uploadModelSource) uploadModelSource.hidden = isHuggingFace;
    if (huggingFaceModelSource) huggingFaceModelSource.hidden = !isHuggingFace;
    if (hfModelId) hfModelId.required = isHuggingFace;
    selectedModelFile = isHuggingFace
      ? null
      : uploadedModelFile || fileInput?.files?.[0] || null;
    updateDeviceRecommendation();
  }

  if (modelSource) {
    modelSource.addEventListener("change", updateModelSource);
    updateModelSource();
  }

  // =========================================================
  // CREATE TARGET DEVICE PANEL
  // =========================================================

  createTargetDevicePanel();
  setDashboardPage("source");

  function createTargetDevicePanel() {
    const existingPanel = document.getElementById("targetDevicePanel");

    if (existingPanel) {
      return;
    }

    const panel = document.createElement("section");

    panel.id = "targetDevicePanel";
    panel.className = "dashboard-page panel device-page";
    panel.dataset.dashboardPage = "device";
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", "tab-device");
    panel.hidden = true;

    panel.innerHTML = `

            <div style="
                font-size:12px;
                letter-spacing:1px;
                color:#d64b2a;
                margin-bottom:6px;
            ">
                05
            </div>

            <h2 style="
                margin:0 0 8px 0;
                font-size:24px;
            ">
                Target device
            </h2>

            <p style="
                margin:0 0 20px 0;
                color:#66736e;
                line-height:1.5;
            ">
                Configure the device where the quantized model will run.
            </p>


            <!-- DEVICE PROFILE -->

            <div style="margin-bottom:18px;">

                <label>

                    <span style="
                        display:block;
                        margin-bottom:6px;
                        font-size:12px;
                        letter-spacing:0.5px;
                    ">
                        DEVICE PROFILE
                    </span>

                    <select
                        id="deviceProfile"
                        style="
                            width:100%;
                            box-sizing:border-box;
                            padding:12px;
                            border:1px solid #cfd6d2;
                            border-radius:3px;
                            background:#fff;
                        "
                    >

                        <option value="custom">
                            Custom device
                        </option>

                        <option value="android_low">
                            Low-end Android
                        </option>

                        <option value="android_mid">
                            Mid-range Android
                        </option>

                        <option value="raspberry_pi">
                            Raspberry Pi / ARM Edge
                        </option>

                        <option value="laptop_4">
                            Basic Laptop - 4 GB
                        </option>

                        <option value="laptop_8">
                            Standard Laptop - 8 GB
                        </option>

                        <option value="laptop_16">
                            Laptop - 16 GB
                        </option>

                        <option value="desktop_gpu">
                            Desktop + NVIDIA GPU
                        </option>

                    </select>

                </label>

            </div>


            <!-- DEVICE CONFIGURATION -->

            <div style="
                display:grid;
                grid-template-columns:
                    repeat(auto-fit,minmax(180px,1fr));
                gap:14px;
            ">


                <!-- RAM -->

                <label>

                    <span style="
                        display:block;
                        margin-bottom:6px;
                        font-size:12px;
                    ">
                        RAM (GB)
                    </span>

                    <input
                        id="targetRam"
                        type="number"
                        min="0.25"
                        step="0.25"
                        value="4"
                        style="
                            width:100%;
                            box-sizing:border-box;
                            padding:12px;
                            border:1px solid #cfd6d2;
                            border-radius:3px;
                        "
                    >

                </label>


                <!-- STORAGE -->

                <label>

                    <span style="
                        display:block;
                        margin-bottom:6px;
                        font-size:12px;
                    ">
                        Storage (GB)
                    </span>

                    <input
                        id="targetStorage"
                        type="number"
                        min="1"
                        step="1"
                        value="32"
                        style="
                            width:100%;
                            box-sizing:border-box;
                            padding:12px;
                            border:1px solid #cfd6d2;
                            border-radius:3px;
                        "
                    >

                </label>


                <!-- CPU -->

                <label>

                    <span style="
                        display:block;
                        margin-bottom:6px;
                        font-size:12px;
                    ">
                        CPU architecture
                    </span>

                    <select
                        id="targetCpu"
                        style="
                            width:100%;
                            box-sizing:border-box;
                            padding:12px;
                            border:1px solid #cfd6d2;
                            border-radius:3px;
                            background:#fff;
                        "
                    >

                        <option value="auto">
                            Auto detect
                        </option>

                        <option value="x86_64">
                            x86_64
                        </option>

                        <option value="arm64">
                            ARM64
                        </option>

                        <option value="arm">
                            ARM
                        </option>

                    </select>

                </label>


                <!-- GPU -->

                <label>

                    <span style="
                        display:block;
                        margin-bottom:6px;
                        font-size:12px;
                    ">
                        GPU
                    </span>

                    <select
                        id="targetGpu"
                        style="
                            width:100%;
                            box-sizing:border-box;
                            padding:12px;
                            border:1px solid #cfd6d2;
                            border-radius:3px;
                            background:#fff;
                        "
                    >

                        <option value="none">
                            None
                        </option>

                        <option value="cuda">
                            NVIDIA CUDA
                        </option>

                        <option value="mobile">
                            Mobile / integrated
                        </option>

                    </select>

                </label>

            </div>


            <!-- RECOMMENDATION -->

            <div
                id="deviceRecommendation"
                style="
                    margin-top:20px;
                    padding:18px;
                    background:#f5f7f5;
                    border-left:3px solid #17211e;
                "
            >

                <strong>
                    Device recommendation
                </strong>

                <p style="
                    margin:8px 0 0 0;
                    color:#66736e;
                ">
                    Select a model to calculate compatibility.
                </p>

            </div>

        `;

    // =====================================================
    // PLACE PANEL
    // =====================================================

    dashboardPages?.appendChild(panel);

    // =====================================================
    // INPUT EVENTS
    // =====================================================

    ["targetRam", "targetStorage", "targetCpu", "targetGpu"].forEach(
      function (id) {
        const element = document.getElementById(id);

        if (!element) {
          return;
        }

        element.addEventListener("input", updateDeviceRecommendation);

        element.addEventListener("change", updateDeviceRecommendation);
      },
    );

    const deviceProfile = document.getElementById("deviceProfile");

    if (deviceProfile) {
      deviceProfile.addEventListener("change", applyDeviceProfile);
    }
  }

  // =========================================================
  // DEVICE PROFILES
  // =========================================================

  function applyDeviceProfile() {
    const profile = document.getElementById("deviceProfile")?.value;

    const ram = document.getElementById("targetRam");

    const storage = document.getElementById("targetStorage");

    const cpu = document.getElementById("targetCpu");

    const gpu = document.getElementById("targetGpu");

    if (!profile || profile === "custom") {
      updateDeviceRecommendation();

      return;
    }

    const profiles = {
      android_low: {
        ram: 2,

        storage: 8,

        cpu: "arm64",

        gpu: "mobile",
      },

      android_mid: {
        ram: 4,

        storage: 32,

        cpu: "arm64",

        gpu: "mobile",
      },

      raspberry_pi: {
        ram: 4,

        storage: 32,

        cpu: "arm64",

        gpu: "none",
      },

      laptop_4: {
        ram: 4,

        storage: 32,

        cpu: "x86_64",

        gpu: "none",
      },

      laptop_8: {
        ram: 8,

        storage: 512,

        cpu: "x86_64",

        gpu: "none",
      },

      laptop_16: {
        ram: 16,

        storage: 512,

        cpu: "x86_64",

        gpu: "none",
      },

      desktop_gpu: {
        ram: 32,

        storage: 1024,

        cpu: "x86_64",

        gpu: "cuda",
      },
    };

    const selected = profiles[profile];

    if (!selected) {
      return;
    }

    ram.value = selected.ram;

    storage.value = selected.storage;

    cpu.value = selected.cpu;

    gpu.value = selected.gpu;

    updateDeviceRecommendation();
  }

  // =========================================================
  // GET TARGET DEVICE
  // =========================================================

  function getTargetDevice() {
    const ram = Number(document.getElementById("targetRam")?.value);

    const storage = Number(document.getElementById("targetStorage")?.value);

    const cpu = document.getElementById("targetCpu")?.value || "auto";

    const gpu = document.getElementById("targetGpu")?.value || "none";

    return {
      ram_gb: ram,

      storage_gb: storage,

      cpu_architecture: cpu,

      gpu_type: gpu,
    };
  }

  // =========================================================
  // GET SELECTED PRECISION
  // =========================================================

  function getSelectedPrecision() {
    if (!quantization) {
      return "int8";
    }

    const value = String(quantization.value || "int8").toLowerCase();

    if (value.includes("fp16") || value.includes("float16")) {
      return "fp16";
    }

    if (
      value.includes("int4") ||
      value.includes("4bit") ||
      value.includes("4-bit")
    ) {
      return "int4";
    }

    if (
      value.includes("int8") ||
      value.includes("8bit") ||
      value.includes("8-bit")
    ) {
      return "int8";
    }

    return value;
  }

  // =========================================================
  // ESTIMATE QUANTIZED MODEL SIZE
  // =========================================================

  function estimateQuantizedSize(modelBytes) {
    if (!modelBytes) {
      return null;
    }

    const precision = getSelectedPrecision();

    /*
     * These are planning estimates.
     *
     * They are NOT a real benchmark.
     *
     * FP16 ≈ 50%
     * INT8 ≈ 28%
     * INT4 ≈ 15%
     */

    let ratio = 0.28;

    if (precision === "fp16") {
      ratio = 0.5;
    } else if (precision === "int4") {
      ratio = 0.15;
    } else if (precision === "int8") {
      ratio = 0.28;
    }

    return modelBytes * ratio;
  }

  // =========================================================
  // ESTIMATE RUNTIME MEMORY
  // =========================================================

  function estimateRuntimeMemory(modelBytes) {
    if (!modelBytes) {
      return null;
    }

    const quantized = estimateQuantizedSize(modelBytes);

    /*
     * Generic planning estimate.
     *
     * Actual runtime memory depends on:
     *
     * - model architecture
     * - inference runtime
     * - activations
     * - batch size
     * - sequence length
     * - hardware
     */

    return quantized * 1.5;
  }

  // =========================================================
  // DEPLOYMENT READINESS
  // =========================================================

  function calculateDeploymentReadiness(
    target,
    originalBytes,
    quantizedBytes,
    estimatedMemoryBytes,
  ) {
    let score = 100;

    const reasons = [];

    const quantizedGb = quantizedBytes / 1024 ** 3;

    const runtimeRamGb = estimatedMemoryBytes / 1024 ** 3;

    const ramRequired = runtimeRamGb * 1.2;

    // -----------------------------------------------------
    // RAM
    // -----------------------------------------------------

    if (target.ram_gb < ramRequired) {
      score -= 35;

      reasons.push({
        type: "warning",

        text: "RAM may be insufficient",
      });
    } else {
      reasons.push({
        type: "success",

        text: "RAM compatible",
      });
    }

    // -----------------------------------------------------
    // STORAGE
    // -----------------------------------------------------

    const storageRequired = quantizedGb * 1.2;

    if (target.storage_gb < storageRequired) {
      score -= 25;

      reasons.push({
        type: "warning",

        text: "Storage may be insufficient",
      });
    } else {
      reasons.push({
        type: "success",

        text: "Storage compatible",
      });
    }

    // -----------------------------------------------------
    // MODEL SIZE
    // -----------------------------------------------------

    if (quantizedGb <= target.ram_gb * 0.25) {
      reasons.push({
        type: "success",

        text: "Quantized model size is suitable",
      });
    } else {
      score -= 10;

      reasons.push({
        type: "warning",

        text: "Model is relatively large for this device",
      });
    }

    // -----------------------------------------------------
    // CPU ARCHITECTURE
    // -----------------------------------------------------

    if (target.cpu_architecture === "arm64") {
      reasons.push({
        type: "success",

        text: "ARM64 edge deployment supported",
      });
    } else if (target.cpu_architecture === "x86_64") {
      reasons.push({
        type: "success",

        text: "x86_64 deployment supported",
      });
    } else {
      score -= 10;

      reasons.push({
        type: "warning",

        text: "CPU architecture should be verified",
      });
    }

    // -----------------------------------------------------
    // GPU
    // -----------------------------------------------------

    if (target.gpu_type === "cuda") {
      reasons.push({
        type: "success",

        text: "CUDA acceleration available",
      });
    } else if (target.gpu_type === "mobile") {
      reasons.push({
        type: "success",

        text: "Mobile/integrated deployment selected",
      });
    } else {
      reasons.push({
        type: "success",

        text: "CPU deployment selected",
      });
    }

    // -----------------------------------------------------
    // PRECISION
    // -----------------------------------------------------

    const precision = getSelectedPrecision();

    if (precision === "int8") {
      reasons.push({
        type: "success",

        text: "INT8 provides a good edge-deployment balance",
      });
    } else if (precision === "int4") {
      reasons.push({
        type: "success",

        text: "INT4 provides aggressive memory reduction",
      });
    } else if (precision === "fp16") {
      reasons.push({
        type: "warning",

        text: "FP16 provides less compression than INT8/INT4",
      });

      score -= 5;
    }

    // -----------------------------------------------------
    // LIMIT SCORE
    // -----------------------------------------------------

    score = Math.max(0, Math.min(100, Math.round(score)));

    // -----------------------------------------------------
    // STATUS
    // -----------------------------------------------------

    let deploymentStatus;
    let statusDescription;

    if (score >= 85) {
      deploymentStatus = "READY";

      statusDescription =
        "The quantized model is suitable for deployment on this device.";
    } else if (score >= 65) {
      deploymentStatus = "READY WITH CAUTION";

      statusDescription =
        "The model may run, but some device limitations should be considered.";
    } else {
      deploymentStatus = "NEEDS OPTIMIZATION";

      statusDescription =
        "Further quantization or model optimization is recommended.";
    }

    return {
      score,

      deploymentStatus,

      statusDescription,

      reasons,
    };
  }

  // =========================================================
  // UPDATE DEVICE RECOMMENDATION
  // =========================================================

  function updateDeviceRecommendation() {
    const recommendation = document.getElementById("deviceRecommendation");

    if (!recommendation) {
      return;
    }

    // -----------------------------------------------------
    // NO MODEL
    // -----------------------------------------------------

    if (!selectedModelFile) {
      const remoteModelId = hfModelId?.value.trim();
      const missingModelMessage =
        modelSource?.value === "huggingface" && remoteModelId
          ? "Repository size and device compatibility will be calculated after download."
          : "Select a model to calculate compatibility.";

      recommendation.innerHTML = `

                <strong>
                    Device recommendation
                </strong>

                <p style="
                    margin:8px 0 0 0;
                    color:#66736e;
                ">
                    ${escapeHtml(missingModelMessage)}
                </p>

            `;

      return;
    }

    // -----------------------------------------------------
    // TARGET
    // -----------------------------------------------------

    const target = getTargetDevice();

    // -----------------------------------------------------
    // VALIDATION
    // -----------------------------------------------------

    if (
      !target.ram_gb ||
      !target.storage_gb ||
      target.ram_gb <= 0 ||
      target.storage_gb <= 0
    ) {
      recommendation.innerHTML = `

                <strong>
                    Device recommendation
                </strong>

                <p style="
                    margin:8px 0 0 0;
                    color:#a33;
                ">
                    Enter valid RAM and storage values.
                </p>

            `;

      return;
    }

    // -----------------------------------------------------
    // ORIGINAL MODEL SIZE
    // -----------------------------------------------------

    const originalBytes = selectedModelFile.size;

    // -----------------------------------------------------
    // ESTIMATES
    // -----------------------------------------------------

    const quantizedBytes = estimateQuantizedSize(originalBytes);

    const estimatedMemoryBytes = estimateRuntimeMemory(originalBytes);

    const quantizedGb = quantizedBytes / 1024 ** 3;

    const estimatedMemoryGb = estimatedMemoryBytes / 1024 ** 3;

    // -----------------------------------------------------
    // SAFE LIMITS
    // -----------------------------------------------------

    const storageLimit = target.storage_gb * 0.8;

    const ramLimit = target.ram_gb * 0.7;

    const storagePass = quantizedGb <= storageLimit;

    const ramPass = estimatedMemoryGb <= ramLimit;

    // -----------------------------------------------------
    // COMPATIBILITY STATUS
    // -----------------------------------------------------

    let status = "Compatible";

    let message = "The selected precision is suitable for this target device.";

    if (!storagePass && !ramPass) {
      status = "Not compatible";

      message =
        "The estimated model requires more storage and runtime memory than the selected device safely provides.";
    } else if (!ramPass) {
      status = "RAM warning";

      message =
        "The model fits storage, but estimated runtime memory is high for the selected RAM.";
    } else if (!storagePass) {
      status = "Storage warning";

      message =
        "The model fits the RAM estimate, but the selected storage is too small.";
    }

    // -----------------------------------------------------
    // DEPLOYMENT READINESS
    // -----------------------------------------------------

    const readiness = calculateDeploymentReadiness(
      target,
      originalBytes,
      quantizedBytes,
      estimatedMemoryBytes,
    );

    const precision = getSelectedPrecision();

    const precisionLabel = precision.toUpperCase();

    // -----------------------------------------------------
    // DISPLAY
    // -----------------------------------------------------

    recommendation.innerHTML = `

            <div style="
                display:flex;
                justify-content:space-between;
                gap:12px;
                flex-wrap:wrap;
            ">

                <strong>
                    Recommended precision:
                    ${precisionLabel}
                </strong>

                <strong>
                    ${escapeHtml(status)}
                </strong>

            </div>


            <div style="
                margin-top:14px;
                display:grid;
                grid-template-columns:
                    repeat(auto-fit,minmax(170px,1fr));
                gap:10px;
                color:#39433f;
            ">

                <div>

                    <small>
                        Original model
                    </small>

                    <br>

                    <strong>
                        ${formatFileSize(originalBytes)}
                    </strong>

                </div>


                <div>

                    <small>
                        Estimated ${precisionLabel} size
                    </small>

                    <br>

                    <strong>
                        ${formatFileSize(quantizedBytes)}
                    </strong>

                </div>


                <div>

                    <small>
                        Estimated runtime RAM
                    </small>

                    <br>

                    <strong>
                        ${estimatedMemoryGb.toFixed(2)}
                        GB
                    </strong>

                </div>


                <div>

                    <small>
                        Target RAM
                    </small>

                    <br>

                    <strong>
                        ${target.ram_gb}
                        GB
                    </strong>

                </div>

            </div>


            <p style="
                margin:14px 0 0 0;
                color:#66736e;
                line-height:1.5;
            ">
                ${escapeHtml(message)}
            </p>


            <!-- =========================================
                 DEPLOYMENT READINESS
                 ========================================= -->

            <div style="
                margin-top:20px;
                padding:20px;
                border:1px solid #d8dedb;
                background:#ffffff;
            ">

                <div style="
                    font-size:12px;
                    letter-spacing:1px;
                    color:#d64b2a;
                    margin-bottom:8px;
                ">
                    DEPLOYMENT READINESS
                </div>


                <div style="
                    display:flex;
                    align-items:center;
                    justify-content:space-between;
                    gap:20px;
                    flex-wrap:wrap;
                ">


                    <!-- SCORE -->

                    <div>

                        <div style="
                            font-size:42px;
                            font-weight:bold;
                            line-height:1;
                        ">

                            ${readiness.score}

                            <span style="
                                font-size:18px;
                                color:#777;
                            ">
                                / 100
                            </span>

                        </div>


                        <div style="
                            margin-top:8px;
                            font-weight:bold;
                        ">

                            ${escapeHtml(readiness.deploymentStatus)}

                        </div>

                    </div>


                    <!-- PROGRESS -->

                    <div style="
                        flex:1;
                        min-width:240px;
                    ">

                        <div style="
                            height:8px;
                            background:#e5e9e7;
                            border-radius:10px;
                            overflow:hidden;
                        ">

                            <div style="
                                width:${readiness.score}%;
                                height:100%;
                                background:#17211e;
                                transition:width .3s ease;
                            "></div>

                        </div>


                        <p style="
                            margin:10px 0 0 0;
                            color:#66736e;
                            line-height:1.5;
                        ">

                            ${escapeHtml(readiness.statusDescription)}

                        </p>

                    </div>

                </div>


                <!-- REASONS -->

                <div style="
                    margin-top:18px;
                    display:grid;
                    gap:8px;
                ">

                    ${readiness.reasons
                      .map(function (reason) {
                        const symbol = reason.type === "success" ? "✓" : "⚠";

                        return `

                                <div style="
                                    font-size:14px;
                                    color:#46514d;
                                ">

                                    <strong>
                                        ${symbol}
                                    </strong>

                                    ${escapeHtml(reason.text)}

                                </div>

                            `;
                      })
                      .join("")}

                </div>

            </div>


            <p style="
                margin:14px 0 0 0;
                color:#8a928e;
                font-size:12px;
            ">

                These are compatibility estimates.
                Actual inference speed and memory depend
                on the model architecture, runtime and
                target hardware.

            </p>

        `;
  }

  // =========================================================
  // FILE SELECTION
  // =========================================================

  if (fileInput) {
    fileInput.addEventListener("change", function () {
      console.log("File input changed");

      const file = fileInput.files[0];
      if (file) {
        console.log("Selected file:", file.name);
        console.log("File size:", file.size);
      }
      setModelFile(file);
    });
  }
  // =========================================================
  // QUANTIZATION FORM
  // =========================================================

  if (form) {
    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      event.stopPropagation();

      console.log("Quantization form submitted");

      const file = selectedModelFile || fileInput?.files?.[0];
      const isHuggingFace = modelSource?.value === "huggingface";
      const repoId = hfModelId?.value.trim() || "";

      // -------------------------------------------------
      // MODEL CHECK
      // -------------------------------------------------

      if (isHuggingFace && !repoId) {
        setDashboardPage("results");
        output.className = "output is-error";
        output.innerHTML = `
          <div class="output-mark">!</div>
          <h3>Enter a Hugging Face model ID</h3>
          <p>Use a repository ID such as organization/model-name.</p>
        `;
        hfModelId?.focus();
        return;
      }

      if (!isHuggingFace && !file) {
        setDashboardPage("results");
        output.className = "output is-error";

        output.innerHTML = `

                        <div class="output-mark">
                            !
                        </div>

                        <h3>
                            Select a model first
                        </h3>

                        <p>
                            Please choose a model file or Hugging Face repository.
                        </p>

                    `;

        return;
      }

      // -------------------------------------------------
      // FILE SIZE
      // -------------------------------------------------

      const MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024;

      if (!isHuggingFace && file.size > MAX_UPLOAD_BYTES) {
        setDashboardPage("results");
        output.className = "output is-error";

        output.innerHTML = `

                        <div class="output-mark">
                            !
                        </div>

                        <h3>
                            Model is too large
                        </h3>

                        <p>
                            Maximum upload size is 20 GB.
                        </p>

                    `;

        return;
      }

      // -------------------------------------------------
      // SETTINGS
      // -------------------------------------------------

      const selectedQuantization = quantization?.value || "int8";

      const selectedDevice = deviceMap?.value || "auto";

      const targetDevice = getTargetDevice();

      console.log("Quantization:", selectedQuantization);

      console.log("Device:", selectedDevice);

      console.log("Target device:", targetDevice);

      updateDeviceRecommendation();

      // -------------------------------------------------
      // UPLOADING
      // -------------------------------------------------

      setDashboardPage("results");
      const modelDescription = isHuggingFace ? repoId : file.name;
      setWorkflowState(
        "working",
        isHuggingFace ? "Downloading..." : "Uploading...",
      );
      if (quantizeButton) {
        quantizeButton.disabled = true;
        quantizeButton.classList.add("is-loading");
        quantizeButton.querySelector(".button-label").textContent =
          isHuggingFace ? "Fetching model..." : "Uploading model...";
      }
      form.setAttribute("aria-busy", "true");

      output.className = "output is-loading";

      output.innerHTML = `

                    <div class="output-mark">
                        ...
                    </div>

                    <h3>${isHuggingFace ? "Fetching model from Hugging Face" : "Uploading model"}</h3>

                    <p>
                        ${escapeHtml(modelDescription)}
                    </p>

                    <p>
                        ${isHuggingFace ? "The server is downloading the repository before quantization." : "Please wait..."}
                    </p>

                `;

      // -------------------------------------------------
      // FORM DATA
      // -------------------------------------------------

      const formData = new FormData();

      formData.append("model_source", isHuggingFace ? "huggingface" : "upload");
      if (isHuggingFace) {
        formData.append("model_id", repoId);
        const revision = document.getElementById("hfRevision")?.value.trim();
        if (revision) formData.append("revision", revision);
      } else {
        formData.append("model", file);
      }

      formData.append("quantization", selectedQuantization);

      formData.append("device_map", selectedDevice);

      if (selectedDatasetFile) {
        formData.append("dataset", selectedDatasetFile);
      }

      // -------------------------------------------------
      // SEND TO FLASK
      // -------------------------------------------------

      try {
        console.log("Sending model request to /api/quantize...");

        const response = await fetch("/api/quantize", {
          method: "POST",
          body: formData,
        });

        console.log("Server status:", response.status);

        let result;

        try {
          result = await response.json();
        } catch (jsonError) {
          const text = await response.text();

          throw new Error(
            "Server returned invalid JSON. " + text.substring(0, 200),
          );
        }

        console.log("Server response:", result);

        // -------------------------------------------------
        // ERROR
        // -------------------------------------------------

        if (!response.ok) {
          throw new Error(result.error || "Server returned an error");
        }

        lastQuantizationResult = result;
        resultsIndicator?.classList.add("has-result");

        setWorkflowState("complete", "Complete");

        // -------------------------------------------------
        // MODEL INFORMATION
        // -------------------------------------------------

        let modelInfoHTML = "";

        if (result.model_detection) {
          const detection = result.model_detection;

          modelInfoHTML = `

                            <p>

                                <strong>
                                    Format:
                                </strong>

                                ${escapeHtml(detection.format || "Unknown")}

                            </p>


                            <p>

                                <strong>
                                    Model:
                                </strong>

                                ${escapeHtml(detection.model_type || "Unknown")}

                            </p>


                            <p>

                                <strong>
                                    Architecture:
                                </strong>

                                ${escapeHtml(
                                  detection.architecture || "Unknown",
                                )}

                            </p>

                        `;
        }

        // -------------------------------------------------
        // DOWNLOAD
        // -------------------------------------------------

        let downloadHTML = "";

        if (result.download_url) {
          downloadHTML = `

                            <p>

                                <a
                                    class="download"
                                    href="${escapeHtml(result.download_url)}"
                                    download
                                >
                                    Download quantized model bundle (.zip) →
                                </a>

                            </p>

                        `;
        }

        // -------------------------------------------------
        // SUCCESS OUTPUT
        // -------------------------------------------------

        output.className = "output is-success";

        output.innerHTML = `

                        <div class="output-mark">
                            OK
                        </div>

                        <h3>
                            Quantization complete
                        </h3>

                        <p>
                            Model processed successfully.
                        </p>


                        <p>

                            <strong>
                                Quantization:
                            </strong>

                            ${escapeHtml(selectedQuantization)}

                        </p>


                        ${modelInfoHTML}


                        ${downloadHTML}

                    `;

        // -------------------------------------------------
        // METRICS
        // -------------------------------------------------

        if (result.metrics) {
          const comparison = document.getElementById("comparison");

          if (comparison) {
            comparison.hidden = false;
          }

          const metrics = result.metrics;

          // ---------------------------------------------
          // SIZE
          // ---------------------------------------------

          if (metrics.size) {
            setText("sizeBefore", formatFileSize(metrics.size.before_bytes));

            setText("sizeAfter", formatFileSize(metrics.size.after_bytes));

            setTextIfExists(
              "sizeReduction",
              formatPercent(metrics.size.reduction_percent),
            );
          }

          // ---------------------------------------------
          // PERFORMANCE
          // ---------------------------------------------

          if (metrics.performance) {
            setText(
              "performanceBefore",
              formatMetric(metrics.performance.before, "ms"),
            );

            setText(
              "performanceAfter",
              formatMetric(metrics.performance.after, "ms"),
            );

            setTextIfExists(
              "performanceChange",
              formatChange(metrics.performance.change_percent, "%"),
            );
          }

          // ---------------------------------------------
          // ACCURACY
          // ---------------------------------------------

          if (metrics.accuracy) {
            setText(
              "accuracyBefore",
              formatMetric(metrics.accuracy.before, "%"),
            );

            setText("accuracyAfter", formatMetric(metrics.accuracy.after, "%"));

            setTextIfExists(
              "accuracyChange",
              formatChange(metrics.accuracy.difference, "%"),
            );

            setTextIfExists("accuracyReason", metrics.accuracy.reason || "");
          }
        }

        // -------------------------------------------------
        // REFRESH DEVICE RECOMMENDATION
        // -------------------------------------------------

        updateDeviceRecommendation();
      } catch (error) {
        // -----------------------------------------------------
        // ERROR HANDLING
        // -----------------------------------------------------

        console.error("Quantization error:", error);

        setDashboardPage("results");
        setWorkflowState("error", "Error");

        output.className = "output is-error";

        output.innerHTML = `

                        <div class="output-mark">
                            !
                        </div>

                        <h3>
                            Quantization failed
                        </h3>

                        <p>
                            ${escapeHtml(error.message)}
                        </p>

                    `;
      } finally {
        form.removeAttribute("aria-busy");
        if (quantizeButton) {
          quantizeButton.disabled = false;
          quantizeButton.classList.remove("is-loading");
          quantizeButton.querySelector(".button-label").textContent =
            "Start quantization";
        }
      }
    });
  }

  // =========================================================
  // DEVICE INFORMATION
  // =========================================================

  async function loadDeviceInfo() {
    try {
      const response = await fetch("/api/device-info");

      if (!response.ok) {
        throw new Error("Device information unavailable");
      }

      const info = await response.json();

      if (info.status === "error") {
        throw new Error(info.error || "Device information unavailable");
      }

      currentDeviceInfo = info.device || null;

      // -------------------------------------------------
      // DEVICE DISPLAY
      // -------------------------------------------------

      if (deviceValue) {
        deviceValue.textContent = info.cuda_available
          ? "CUDA"
          : currentDeviceInfo?.device_type || "CPU";
      }

      // -------------------------------------------------
      // MEMORY
      // -------------------------------------------------

      if (info.memory_total) {
        if (memoryValue) {
          memoryValue.textContent = formatFileSize(info.memory_total);
        }

        const currentRam = info.memory_total / 1024 ** 3;

        if (Number.isFinite(currentRam)) {
          const ramInput = document.getElementById("ram");

          if (ramInput && !ramInput.matches(":focus")) {
            ramInput.value = currentRam.toFixed(2);
          }
        }
      }

      // -------------------------------------------------
      // CPU
      // -------------------------------------------------

      if (currentDeviceInfo?.cpu_architecture) {
        const cpuSelect = document.getElementById("cpuArchitecture");

        if (cpuSelect) {
          const cpuValue = currentDeviceInfo.cpu_architecture;

          const option = Array.from(cpuSelect.options).find(function (item) {
            return item.value === cpuValue;
          });

          if (option) {
            cpuSelect.value = cpuValue;
          }
        }
      }

      // -------------------------------------------------
      // GPU
      // -------------------------------------------------

      if (info.cuda_available) {
        const gpuSelect = document.getElementById("gpu");

        if (gpuSelect) {
          const cudaOption = Array.from(gpuSelect.options).find(
            function (item) {
              return item.value === "cuda";
            },
          );

          if (cudaOption) {
            gpuSelect.value = "cuda";
          }
        }
      }

      updateDeviceRecommendation();
    } catch (error) {
      console.warn("Device information error:", error);

      if (deviceValue) {
        deviceValue.textContent = "CPU";
      }
    }
  }

  // =========================================================
  // DEVICE PROFILE CHANGE
  // =========================================================

  if (deviceProfile) {
    deviceProfile.addEventListener("change", function () {
      console.log("Device profile changed:", deviceProfile.value);

      applyDeviceProfile();

      updateDeviceRecommendation();
    });
  }

  // =========================================================
  // TARGET DEVICE INPUTS
  // =========================================================

  const targetInputs = ["ram", "storage", "cpuArchitecture", "gpu"];

  targetInputs.forEach(function (id) {
    const element = document.getElementById(id);

    if (!element) {
      return;
    }

    element.addEventListener("input", function () {
      updateDeviceRecommendation();
    });

    element.addEventListener("change", function () {
      updateDeviceRecommendation();
    });
  });

  // =========================================================
  // PRECISION CHANGE
  // =========================================================

  if (quantization) {
    quantization.addEventListener("change", function () {
      console.log("Precision changed:", quantization.value);

      updateDeviceRecommendation();
    });
  }

  // =========================================================
  // DEVICE MAP CHANGE
  // =========================================================

  if (deviceMap) {
    deviceMap.addEventListener("change", function () {
      console.log("Device map changed:", deviceMap.value);

      updateDeviceRecommendation();
    });
  }

  // =========================================================
  // INITIAL DEVICE PROFILE
  // =========================================================

  applyDeviceProfile();

  // =========================================================
  // INITIAL DEVICE INFORMATION
  // =========================================================

  loadDeviceInfo();

  // =========================================================
  // INITIAL RECOMMENDATION
  // =========================================================

  updateDeviceRecommendation();

  console.log("Quantization Studio initialized successfully.");

  // =========================================================
  // DEVICE INFORMATION
  // =========================================================

  async function loadDeviceInfo() {
    try {
      const response = await fetch("/api/device-info");

      if (!response.ok) {
        throw new Error("Device information unavailable");
      }

      const info = await response.json();

      if (info.status === "error") {
        throw new Error(info.error || "Device information unavailable");
      }

      currentDeviceInfo = info.device || null;

      // -------------------------------------------------
      // DEVICE DISPLAY
      // -------------------------------------------------

      if (deviceValue) {
        deviceValue.textContent = info.cuda_available
          ? "CUDA"
          : currentDeviceInfo?.device_type || "CPU";
      }

      // -------------------------------------------------
      // MEMORY
      // -------------------------------------------------

      if (info.memory_total) {
        if (memoryValue) {
          memoryValue.textContent = formatFileSize(info.memory_total);
        }

        const currentRam = info.memory_total / 1024 ** 3;

        const targetRam = document.getElementById("targetRam");

        if (targetRam && currentRam) {
          const profile = document.getElementById("deviceProfile")?.value;

          /*
           * Only update the target RAM when
           * the user is using the custom profile.
           */

          if (!profile || profile === "custom") {
            targetRam.value = currentRam.toFixed(2);
          }
        }
      } else {
        if (memoryValue) {
          memoryValue.textContent = "Unavailable";
        }
      }

      // -------------------------------------------------
      // CPU ARCHITECTURE
      // -------------------------------------------------

      const targetCpu = document.getElementById("targetCpu");

      if (targetCpu && currentDeviceInfo?.architecture) {
        const architecture = String(
          currentDeviceInfo.architecture,
        ).toLowerCase();

        const profile = document.getElementById("deviceProfile")?.value;

        /*
         * Don't override a selected device profile.
         */

        if (!profile || profile === "custom") {
          if (
            architecture.includes("amd64") ||
            architecture.includes("x86_64")
          ) {
            targetCpu.value = "x86_64";
          } else if (
            architecture.includes("arm64") ||
            architecture.includes("aarch64")
          ) {
            targetCpu.value = "arm64";
          }
        }
      }

      updateDeviceRecommendation();
    } catch (error) {
      console.error("Device information error:", error);

      currentDeviceInfo = null;

      if (deviceValue) {
        deviceValue.textContent = "Offline";
      }

      if (memoryValue) {
        memoryValue.textContent = "Unavailable";
      }

      updateDeviceRecommendation();
    }
  }

  // =========================================================
  // HELPER: SET TEXT
  // =========================================================

  function setText(id, value) {
    const element = document.getElementById(id);

    if (element) {
      element.textContent = value;
    }
  }

  // =========================================================
  // HELPER: SET TEXT IF EXISTS
  // =========================================================

  function setTextIfExists(id, value) {
    const element = document.getElementById(id);

    if (element) {
      element.textContent = value;
    }
  }

  // =========================================================
  // FORMAT METRIC
  // =========================================================

  function formatMetric(value, unit) {
    if (
      value === null ||
      value === undefined ||
      value === "" ||
      value === "Not measured"
    ) {
      return "Not measured";
    }

    if (typeof value === "number" && !isNaN(value)) {
      return `${value} ${unit}`;
    }

    return String(value);
  }

  // =========================================================
  // FORMAT PERCENT
  // =========================================================

  function formatPercent(value) {
    if (value === null || value === undefined || value === "" || isNaN(value)) {
      return "Not available";
    }

    return `${Number(value).toFixed(2)}%`;
  }

  // =========================================================
  // FORMAT CHANGE
  // =========================================================

  function formatChange(value, unit) {
    if (value === null || value === undefined || value === "" || isNaN(value)) {
      return "Not measured";
    }

    const number = Number(value);

    const sign = number > 0 ? "+" : "";

    return `${sign}${number.toFixed(2)}${unit}`;
  }

  // =========================================================
  // FORMAT FILE SIZE
  // =========================================================

  function formatFileSize(bytes) {
    if (bytes === null || bytes === undefined || isNaN(bytes)) {
      return "Unknown";
    }

    if (bytes < 1024) {
      return bytes + " B";
    }

    if (bytes < 1024 * 1024) {
      return (bytes / 1024).toFixed(1) + " KB";
    }

    if (bytes < 1024 * 1024 * 1024) {
      return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }

    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + " GB";
  }

  // =========================================================
  // ESCAPE HTML
  // =========================================================

  function escapeHtml(value) {
    if (value === null || value === undefined) {
      return "";
    }

    return String(value)
      .replace(/&/g, "&amp;")

      .replace(/</g, "&lt;")

      .replace(/>/g, "&gt;")

      .replace(/"/g, "&quot;")

      .replace(/'/g, "&#039;");
  }

  // =========================================================
  // START DEVICE CHECK
  // =========================================================

  loadDeviceInfo();
});
