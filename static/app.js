document.addEventListener("DOMContentLoaded", () => {
    // -- ELEMENT REFS --
    const uploadZone        = document.getElementById("upload-zone");
    const fileInput         = document.getElementById("file-input");
    const liveFeed          = document.getElementById("live-feed");
    const statusBadge       = document.getElementById("status-badge");
    const progressContainer = document.getElementById("progress-container");
    const progressBarFill   = document.getElementById("progress-bar-fill");
    const frameCounter      = document.getElementById("frame-counter");
    const percentageCounter = document.getElementById("percentage-counter");

    // Stats
    const signalBox      = document.getElementById("signal-box-status");
    const signalTimer    = document.getElementById("signal-timer-val");
    const densityLeft    = document.getElementById("density-left");
    const densityRight   = document.getElementById("density-right");
    const speedLeft      = document.getElementById("speed-left");
    const speedRight     = document.getElementById("speed-right");
    const totalProcessed = document.getElementById("total-processed-count");

    // Vehicle class count cells
    const counts = {
        car:        document.getElementById("count-car"),
        bus:        document.getElementById("count-bus"),
        truck:      document.getElementById("count-truck"),
        motorcycle: document.getElementById("count-motorcycle"),
        person:     document.getElementById("count-person")
    };

    // Mode tabs
    const tabDemo   = document.getElementById("tab-demo");
    const tabUpload = document.getElementById("tab-upload");

    // Results screen
    const resultsScreen     = document.getElementById("results-screen");
    const resultsTotalCount = document.getElementById("results-total-count");
    const btnAnalyzeAnother = document.getElementById("btn-analyze-another");

    // Error screen
    const errorScreen      = document.getElementById("error-screen");
    const errorMessageText = document.getElementById("error-message-text");
    const btnErrorReturn   = document.getElementById("btn-error-return");

    // Chatbot
    const chatTriggerBtn = document.getElementById("chat-trigger-btn");
    const chatDrawer     = document.getElementById("chat-drawer");
    const chatCloseBtn   = document.getElementById("chat-close-btn");
    const chatHistory    = document.getElementById("chat-history");
    const chatInput      = document.getElementById("chat-input");
    const chatSendBtn    = document.getElementById("chat-send-btn");

    // -- STATE --
    let statusInterval    = null;
    let densityTimeSeries = [];
    let chartBreakdown    = null;
    let chartDensity      = null;

    // -- VIEW MANAGEMENT --
    // Exactly one of the four viewport children is visible at any time.
    function showView(id) {
        uploadZone.style.display    = "none";
        liveFeed.style.display      = "none";
        resultsScreen.style.display = "none";
        errorScreen.style.display   = "none";

        if (id === "upload-zone")    uploadZone.style.display    = "flex";
        if (id === "live-feed")      liveFeed.style.display      = "block";
        if (id === "results-screen") resultsScreen.style.display = "flex";
        if (id === "error-screen")   errorScreen.style.display   = "flex";
    }

    function setTabsDisabled(disabled) {
        tabDemo.disabled   = disabled;
        tabUpload.disabled = disabled;
    }

    function setActiveTab(tabId) {
        tabDemo.classList.toggle("active",   tabId === "tab-demo");
        tabUpload.classList.toggle("active", tabId === "tab-upload");
    }

    // -- IDLE STATE --
    function resetToIdle() {
        if (statusInterval) clearInterval(statusInterval);
        showView("upload-zone");
        progressContainer.style.display = "none";
        progressBarFill.style.width     = "0%";
        percentageCounter.textContent   = "0%";
        frameCounter.textContent        = "Frame 0 / 0";
        statusBadge.textContent = "Idle";
        statusBadge.className   = "badge";
        setTabsDisabled(false);
    }

    // -- MODE TAB HANDLERS --
    tabDemo.addEventListener("click", () => {
        setActiveTab("tab-demo");
        startDemo();
    });

    tabUpload.addEventListener("click", () => {
        setActiveTab("tab-upload");
        resetToIdle();
    });

    // -- RESULTS & ERROR BUTTON HANDLERS --
    btnAnalyzeAnother.addEventListener("click", () => {
        setActiveTab("tab-upload");
        resetToIdle();
    });

    btnErrorReturn.addEventListener("click", () => {
        setActiveTab("tab-upload");
        resetToIdle();
    });

    // -- UPLOAD HANDLERS --
    uploadZone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) handleUpload(e.target.files[0]);
    });

    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.classList.add("dragover");
    });

    uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("dragover");
    });

    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) handleUpload(e.dataTransfer.files[0]);
    });

    function handleUpload(file) {
        const formData = new FormData();
        formData.append("file", file);

        statusBadge.textContent = "Uploading...";
        statusBadge.className   = "badge processing";

        // XHR -- no default timeout, safe for large video files (100 MB+)
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener("progress", (e) => {
            if (e.lengthComputable) {
                const pct = Math.round((e.loaded / e.total) * 100);
                statusBadge.textContent         = `Uploading ${pct}%`;
                progressBarFill.style.width     = `${pct}%`;
                percentageCounter.textContent   = `${pct}%`;
                frameCounter.textContent        = "Uploading...";
                progressContainer.style.display = "flex";
            }
        });

        xhr.addEventListener("load", () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                progressBarFill.style.width   = "0%";
                percentageCounter.textContent = "0%";
                frameCounter.textContent      = "Frame 0 / 0";

                setActiveTab("tab-upload");
                setTabsDisabled(true);
                showView("live-feed");
                progressContainer.style.display = "flex";
                statusBadge.textContent = "Processing";
                statusBadge.className   = "badge processing";
                startStatusPolling();
            } else {
                let msg = "Upload failed. Verify file format.";
                try { msg = JSON.parse(xhr.responseText).detail || msg; } catch (_) {}
                statusBadge.textContent = "Upload Failed";
                statusBadge.className   = "badge";
                progressContainer.style.display = "none";
                alert(msg);
            }
        });

        xhr.addEventListener("error", () => {
            statusBadge.textContent = "Upload Failed";
            statusBadge.className   = "badge";
            progressContainer.style.display = "none";
            alert("Network error during upload. Check that the server is running.");
        });

        xhr.open("POST", "/api/upload");
        xhr.send(formData);
    }

    // -- POLLING STATUS LOOP --
    function startStatusPolling() {
        if (statusInterval) clearInterval(statusInterval);
        densityTimeSeries = [];

        statusInterval = setInterval(() => {
            fetch("/api/status")
            .then(res => res.json())
            .then(state => {
                updateDashboard(state);

                // Collect density sample for the results line chart
                if (state.progress > 0) {
                    densityTimeSeries.push({
                        pct:   state.progress,
                        total: state.left_density + state.right_density
                    });
                }

                if (state.status === "completed") {
                    clearInterval(statusInterval);
                    setTabsDisabled(false);
                    showResults(state);
                } else if (state.status === "failed") {
                    clearInterval(statusInterval);
                    setTabsDisabled(false);
                    showError(state.error_message || "An unknown error occurred during analysis.");
                }
            })
            .catch(err => { console.error("Polling error:", err); });
        }, 300);
    }

    // -- DASHBOARD UPDATE --
    function updateDashboard(state) {
        if (state.frame_b64) {
            liveFeed.src = `data:image/jpeg;base64,${state.frame_b64}`;
        }
        progressBarFill.style.width   = `${state.progress}%`;
        percentageCounter.textContent = `${state.progress}%`;
        frameCounter.textContent      = `Frame ${state.current_frame} / ${state.total_frames}`;

        signalBox.textContent   = state.signal_state;
        signalBox.className     = `signal-box ${state.signal_state}`;
        signalTimer.textContent = `${state.signal_timer.toFixed(1)}s`;

        densityLeft.textContent  = state.left_density;
        densityRight.textContent = state.right_density;
        speedLeft.textContent    = state.avg_speed_left.toFixed(1);
        speedRight.textContent   = state.avg_speed_right.toFixed(1);
        totalProcessed.textContent = state.processed_count;

        counts.car.textContent        = state.car_count;
        counts.bus.textContent        = state.bus_count;
        counts.truck.textContent      = state.truck_count;
        counts.motorcycle.textContent = state.motorcycle_count;
        counts.person.textContent     = state.person_count;
    }

    // -- RESULTS SCREEN --
    function showResults(finalState) {
        showView("results-screen");
        statusBadge.textContent         = "Completed";
        statusBadge.className           = "badge completed";
        progressContainer.style.display = "none";
        resultsTotalCount.textContent   = finalState.processed_count;

        // Destroy stale instances before re-rendering (handles demo replay)
        if (chartBreakdown) { chartBreakdown.destroy(); chartBreakdown = null; }
        if (chartDensity)   { chartDensity.destroy();   chartDensity   = null; }

        const gridColor = "rgba(51, 65, 85, 0.6)";
        const textColor = "#94a3b8";

        // 1. Class breakdown -- horizontal bar chart
        chartBreakdown = new Chart(document.getElementById("chart-breakdown"), {
            type: "bar",
            data: {
                labels: ["Cars", "Buses", "Trucks", "Motorcycles", "Pedestrians"],
                datasets: [{
                    data: [
                        finalState.car_count,
                        finalState.bus_count,
                        finalState.truck_count,
                        finalState.motorcycle_count,
                        finalState.person_count
                    ],
                    backgroundColor: [
                        "rgba(99, 102, 241, 0.8)",
                        "rgba(16, 185, 129, 0.8)",
                        "rgba(245, 158, 11, 0.8)",
                        "rgba(239, 68, 68, 0.8)",
                        "rgba(148, 163, 184, 0.8)"
                    ],
                    borderRadius: 4,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                indexAxis: "y",
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: gridColor }, ticks: { color: textColor }, beginAtZero: true },
                    y: { grid: { color: gridColor }, ticks: { color: textColor } }
                }
            }
        });

        // 2. Density over time -- area line chart (down-sampled to 60 pts max)
        const raw  = densityTimeSeries;
        const step = Math.max(1, Math.floor(raw.length / 60));
        const pts  = raw.filter((_, i) => i % step === 0);

        chartDensity = new Chart(document.getElementById("chart-density"), {
            type: "line",
            data: {
                labels: pts.map(d => `${d.pct.toFixed(0)}%`),
                datasets: [{
                    data:            pts.map(d => d.total),
                    borderColor:     "#10b981",
                    backgroundColor: "rgba(16, 185, 129, 0.12)",
                    fill:        true,
                    tension:     0.35,
                    pointRadius: 0,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: gridColor }, ticks: { color: textColor, maxTicksLimit: 7 } },
                    y: {
                        grid: { color: gridColor }, ticks: { color: textColor }, beginAtZero: true,
                        title: { display: true, text: "Vehicles in frame", color: textColor, font: { size: 10 } }
                    }
                }
            }
        });
    }

    // -- ERROR SCREEN --
    function showError(message) {
        showView("error-screen");
        errorMessageText.textContent    = message;
        statusBadge.textContent         = "Failed";
        statusBadge.className           = "badge";
        progressContainer.style.display = "none";
        setTabsDisabled(false);
    }

    // -- DEMO AUTO-PLAY --
    function startDemo() {
        fetch("/api/demo", { method: "POST" })
        .then(res => {
            if (!res.ok) {
                console.warn("Demo video not available -- falling back to upload mode.");
                setActiveTab("tab-upload");
                resetToIdle();
                return;
            }
            setActiveTab("tab-demo");
            setTabsDisabled(true);
            showView("live-feed");
            progressContainer.style.display = "flex";
            progressBarFill.style.width     = "0%";
            percentageCounter.textContent   = "0%";
            frameCounter.textContent        = "Frame 0 / 0";
            statusBadge.textContent = "Demo Mode";
            statusBadge.className   = "badge demo";
            startStatusPolling();
        })
        .catch(err => {
            console.warn("Demo start failed (server may still be booting):", err);
            setActiveTab("tab-upload");
            resetToIdle();
        });
    }

    // -- CHATBOT UI --
    chatTriggerBtn.addEventListener("click", () => {
        chatDrawer.classList.toggle("open");
        chatInput.focus();
    });

    chatCloseBtn.addEventListener("click", () => { chatDrawer.classList.remove("open"); });
    chatSendBtn.addEventListener("click", sendChatMessage);
    chatInput.addEventListener("keypress", (e) => { if (e.key === "Enter") sendChatMessage(); });

    function sendChatMessage() {
        const text = chatInput.value.trim();
        if (!text) return;
        appendMessage(text, "user-message");
        chatInput.value = "";
        fetch("/api/chat", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ query: text })
        })
        .then(res => res.json())
        .then(data => { appendMessage(data.response, "bot-message"); })
        .catch(() => { appendMessage("Sorry, I encountered an error. Please verify the backend API is running.", "bot-message"); });
    }

    function appendMessage(text, className) {
        const msgDiv = document.createElement("div");
        msgDiv.className   = `chat-message ${className}`;
        msgDiv.textContent = text;
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    // -- BOOT: auto-run demo on every page load --
    startDemo();
});
