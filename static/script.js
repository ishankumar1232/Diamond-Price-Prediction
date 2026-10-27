/**
 * DiamondIQ — Production Client Application Logic
 * Clean, robust implementation for mode switching, form validation, and What-If simulator.
 */

document.addEventListener("DOMContentLoaded", function () {

    // =========================================
    // 1. DATASET MODE SWITCHING
    // =========================================
    const modeButtons    = document.querySelectorAll("[data-mode]");
    const standardFields = document.getElementById("standard-fields");
    const detailedFields = document.getElementById("detailed-fields");
    const modeInput      = document.getElementById("dataset-mode");
    const formTitle      = document.getElementById("form-title");
    const simulatorPanel = document.getElementById("simulator") || document.querySelector(".simulator-card");

    function setFieldsEnabled(container, enabled) {
        if (!container) return;
        container.querySelectorAll("input, select, textarea").forEach(function (el) {
            el.disabled = !enabled;
            el.required = enabled;
        });
    }

    function selectMode(mode) {
        const isStandard = mode === "standard";

        if (standardFields) {
            standardFields.classList.toggle("hidden-fields", !isStandard);
            setFieldsEnabled(standardFields, isStandard);
        }

        if (detailedFields) {
            detailedFields.classList.toggle("hidden-fields", isStandard);
            setFieldsEnabled(detailedFields, !isStandard);
        }

        if (modeInput) {
            modeInput.value = mode;
        }

        if (formTitle) {
            formTitle.textContent = isStandard
                ? "Standard Diamond Details"
                : "Detailed Diamond Details";
        }

        modeButtons.forEach(function (btn) {
            const active = btn.dataset.mode === mode;
            btn.classList.toggle("active", active);
            btn.setAttribute("aria-selected", String(active));
            btn.setAttribute("aria-pressed", String(active));
        });

        // Show the What-If simulator only for the standard dataset
        if (simulatorPanel) {
            simulatorPanel.hidden = !isStandard;
        }
    }

    // Mode buttons navigation and toggle
    modeButtons.forEach(function (btn) {
        btn.addEventListener("click", function () {
            const selected = btn.dataset.mode;
            if (selected === "standard" || selected === "detailed") {
                // If on /prediction or /predict, navigate to clear stale results
                if (window.location.pathname === "/prediction" || window.location.pathname === "/predict") {
                    window.location.href = "/prediction?mode=" + encodeURIComponent(selected);
                } else {
                    selectMode(selected);
                }
            }
        });
    });

    // Apply the server-rendered initial mode
    const initialMode = modeInput && modeInput.value === "detailed"
        ? "detailed"
        : "standard";

    selectMode(initialMode);


    // =========================================
    // 2. WHAT-IF CARAT SIMULATOR
    // =========================================
    const caratInput       = document.getElementById("carat");
    const slider           = document.getElementById("simulator-carat");
    const valueLabel       = document.getElementById("simulator-carat-value");
    const applyButton      = document.getElementById("apply-simulator");
    const baselinePriceEl  = document.getElementById("baseline-price");
    const baselineCaratEl  = document.getElementById("baseline-carat");
    const presetButtons    = document.querySelectorAll("[data-preset]");

    function updateSliderLabel(value) {
        if (!valueLabel) return;
        const n = Number(value);
        valueLabel.textContent = Number.isFinite(n) ? n.toFixed(2) : "1.00";
    }

    function syncSliderFromInput() {
        if (!caratInput || !slider) return;
        const carat = Number(caratInput.value);
        if (Number.isFinite(carat) && carat >= 0.1 && carat <= 5) {
            slider.value = String(carat);
            updateSliderLabel(carat);
        }
    }

    if (slider) {
        slider.addEventListener("input", function () {
            updateSliderLabel(slider.value);
        });
    }

    if (caratInput) {
        caratInput.addEventListener("input", syncSliderFromInput);
    }

    // Quick Carat Preset Chips
    presetButtons.forEach(function (presetBtn) {
        presetBtn.addEventListener("click", function () {
            const presetVal = Number(presetBtn.dataset.preset);
            if (Number.isFinite(presetVal) && slider && caratInput) {
                slider.value = String(presetVal);
                updateSliderLabel(presetVal);
                caratInput.value = presetVal.toFixed(2);
                caratInput.dispatchEvent(new Event("input", { bubbles: true }));
            }
        });
    });

    const predictionForm  = document.getElementById("prediction-form");
    const simulatorStatus = document.getElementById("simulator-status");

    if (applyButton && slider && caratInput && predictionForm) {
        applyButton.addEventListener("click", function () {
            const selectedCarat = Number(slider.value);

            if (!Number.isFinite(selectedCarat) || selectedCarat < 0.1 || selectedCarat > 5.0) {
                alert("Please select a valid carat between 0.10 and 5.00.");
                return;
            }

            // Capture first visible prediction as comparison baseline if not set yet
            if (baselinePriceEl && baselineCaratEl && !baselinePriceEl.value.trim()) {
                const usdEl = document.querySelector(".result-price-usd") || document.querySelector(".result-price");
                if (usdEl) {
                    const rawText = usdEl.textContent.replace(/[^0-9.]/g, "").trim();
                    const originalPrice = parseFloat(rawText);
                    const originalCarat = parseFloat(caratInput.value);
                    if (Number.isFinite(originalPrice) && Number.isFinite(originalCarat)) {
                        baselinePriceEl.value = originalPrice.toFixed(10);
                        baselineCaratEl.value = originalCarat.toFixed(2);
                    }
                }
            }

            caratInput.value = selectedCarat.toFixed(2);
            updateSliderLabel(selectedCarat);
            caratInput.dispatchEvent(new Event("input", { bubbles: true }));

            // Show clear loading message
            applyButton.textContent = "Estimating Price...";
            if (simulatorStatus) {
                simulatorStatus.textContent = "Calculating valuation for " + selectedCarat.toFixed(2) + " ct...";
            }

            predictionForm.submit();
        });
    }

    syncSliderFromInput();


    // =========================================
    // 3. PROGRESS BARS ANIMATION
    // =========================================
    document.querySelectorAll(".impact-bar[data-width], .confidence-bar-fill[data-width]").forEach(function (bar) {
        const parsed = Number.parseFloat(bar.dataset.width);
        const width  = Number.isFinite(parsed) ? Math.min(100, Math.max(0, parsed)) : 0;
        requestAnimationFrame(function () {
            bar.style.width = width + "%";
        });
    });


    // =========================================
    // 4. FORM VALIDATION
    // =========================================
    if (predictionForm) {
        predictionForm.addEventListener("submit", function (event) {
            const activeMode      = modeInput ? modeInput.value : "standard";
            const activeContainer = activeMode === "detailed" ? detailedFields : standardFields;

            if (!activeContainer) return;

            const fields = activeContainer.querySelectorAll("input[required], select[required], textarea[required]");

            for (const field of fields) {
                if (field.disabled) continue;
                if (!field.checkValidity()) {
                    event.preventDefault();
                    field.reportValidity();
                    field.focus();
                    return;
                }
            }
        });
    }

});
