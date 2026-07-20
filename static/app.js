document.addEventListener("DOMContentLoaded", () => {
    // UI Elements
    const uploadZone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");
    const liveFeed = document.getElementById("live-feed");
    const statusBadge = document.getElementById("status-badge");
    const progressContainer = document.getElementById("progress-container");
    const progressBarFill = document.getElementById("progress-bar-fill");
    const frameCounter = document.getElementById("frame-counter");
    const percentageCounter = document.getElementById("percentage-counter");
    
    // Stats Elements
    const signalBox = document.getElementById("signal-box-status");
    const signalTimer = document.getElementById("signal-timer-val");
    const densityLeft = document.getElementById("density-left");
    const densityRight = document.getElementById("density-right");
    const speedLeft = document.getElementById("speed-left");
    const speedRight = document.getElementById("speed-right");
    const totalProcessed = document.getElementById("total-processed-count");
    
    // Class Counts
    const counts = {
        car: document.getElementById("count-car"),
        bus: document.getElementById("count-bus"),
        truck: document.getElementById("count-truck"),
        motorcycle: document.getElementById("count-motorcycle"),
        person: document.getElementById("count-person")
    };

    // Chatbot Elements
    const chatTriggerBtn = document.getElementById("chat-trigger-btn");
    const chatDrawer = document.getElementById("chat-drawer");
    const chatCloseBtn = document.getElementById("chat-close-btn");
    const chatHistory = document.getElementById("chat-history");
    const chatInput = document.getElementById("chat-input");
    const chatSendBtn = document.getElementById("chat-send-btn");

    let statusInterval = null;

    // --- UPLOAD HANDLERS ---
    uploadZone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleUpload(e.target.files[0]);
        }
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
        if (e.dataTransfer.files.length > 0) {
            handleUpload(e.dataTransfer.files[0]);
        }
    });

    function handleUpload(file) {
        const formData = new FormData();
        formData.append("file", file);

        statusBadge.textContent = "Uploading...";
        statusBadge.className = "badge processing";

        // Use XHR instead of fetch — XHR has no default timeout so large
        // video files (100 MB+) won't fail with "Failed to fetch".
        const xhr = new XMLHttpRequest();

        // Show real byte-level upload progress while the file is sending
        xhr.upload.addEventListener("progress", (e) => {
            if (e.lengthComputable) {
                const pct = Math.round((e.loaded / e.total) * 100);
                statusBadge.textContent = `Uploading ${pct}%`;
                progressBarFill.style.width = `${pct}%`;
                percentageCounter.textContent = `${pct}%`;
                frameCounter.textContent = `Uploading…`;
                progressContainer.style.display = "flex";
            }
        });

        xhr.addEventListener("load", () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                console.log("Upload Success");
                // Reset progress bar before pipeline starts
                progressBarFill.style.width = "0%";
                percentageCounter.textContent = "0%";
                frameCounter.textContent = "Frame 0 / 0";

                uploadZone.style.display = "none";
                liveFeed.style.display = "block";
                progressContainer.style.display = "flex";
                startStatusPolling();
            } else {
                let msg = "Upload failed. Verify file format.";
                try { msg = JSON.parse(xhr.responseText).detail || msg; } catch (_) {}
                statusBadge.textContent = "Upload Failed";
                statusBadge.className = "badge";
                progressContainer.style.display = "none";
                alert(msg);
            }
        });

        xhr.addEventListener("error", () => {
            statusBadge.textContent = "Upload Failed";
            statusBadge.className = "badge";
            progressContainer.style.display = "none";
            alert("Network error during upload. Check that the server is running.");
        });

        xhr.open("POST", "/api/upload");
        xhr.send(formData);
    }

    // --- POLLING STATUS LOOP ---
    function startStatusPolling() {
        if (statusInterval) clearInterval(statusInterval);
        
        statusBadge.textContent = "Processing";
        statusBadge.className = "badge processing";

        statusInterval = setInterval(() => {
            fetch("/api/status")
            .then(res => res.json())
            .then(state => {
                updateDashboard(state);
                
                if (state.status === "completed") {
                    clearInterval(statusInterval);
                    statusBadge.textContent = "Completed";
                    statusBadge.className = "badge completed";
                } else if (state.status === "failed") {
                    clearInterval(statusInterval);
                    statusBadge.textContent = "Failed";
                    statusBadge.className = "badge";
                    alert("Analysis failed: " + state.error_message);
                }
            })
            .catch(err => {
                console.error("Polling error:", err);
            });
        }, 300); // Poll every 300ms
    }

    function updateDashboard(state) {
        // 1. Update Video Frame Feed from base64 data in the status response
        //    (no file I/O, no Windows file-lock race condition)
        if (state.frame_b64) {
            liveFeed.src = `data:image/jpeg;base64,${state.frame_b64}`;
        }

        // 2. Update Progress Bar
        progressBarFill.style.width = `${state.progress}%`;
        percentageCounter.textContent = `${state.progress}%`;
        frameCounter.textContent = `Frame ${state.current_frame} / ${state.total_frames}`;

        // 3. Update Signal Controller Widget
        signalBox.textContent = state.signal_state;
        signalBox.className = `signal-box ${state.signal_state}`;
        signalTimer.textContent = `${state.signal_timer.toFixed(1)}s`;

        // 4. Update Metric cards
        densityLeft.textContent = state.left_density;
        densityRight.textContent = state.right_density;
        
        speedLeft.textContent = state.avg_speed_left.toFixed(1);
        speedRight.textContent = state.avg_speed_right.toFixed(1);
        
        totalProcessed.textContent = state.processed_count;

        // 5. Update Table counts
        counts.car.textContent = state.car_count;
        counts.bus.textContent = state.bus_count;
        counts.truck.textContent = state.truck_count;
        counts.motorcycle.textContent = state.motorcycle_count;
        counts.person.textContent = state.person_count;
    }

    // --- CHATBOT UI ACTIONS ---
    chatTriggerBtn.addEventListener("click", () => {
        chatDrawer.classList.toggle("open");
        chatInput.focus();
    });

    chatCloseBtn.addEventListener("click", () => {
        chatDrawer.classList.remove("open");
    });

    chatSendBtn.addEventListener("click", sendChatMessage);
    chatInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendChatMessage();
    });

    function sendChatMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        // Render User Message
        appendMessage(text, "user-message");
        chatInput.value = "";

        // Send to API
        fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ query: text })
        })
        .then(res => res.json())
        .then(data => {
            // Render Bot Response
            appendMessage(data.response, "bot-message");
        })
        .catch(err => {
            console.error("Chat error:", err);
            appendMessage("Sorry, I encountered an error. Please verify the backend API is running.", "bot-message");
        });
    }

    function appendMessage(text, className) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `chat-message ${className}`;
        msgDiv.textContent = text;
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight; // Auto scroll
    }
});
