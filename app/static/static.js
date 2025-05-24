document.getElementById("analyzeBtn").addEventListener("click", async () => {
    const inputType = document.getElementById("inputType").value;
    const inputData = document.getElementById("inputData").value.trim();

    if (!inputData) {
        alert("يرجى إدخال نص أو رابط للتحليل");
        return;
    }

    const payload = inputType === "text" ? { input_text: inputData } : { input_url: inputData };

    document.getElementById("result").textContent = "جاري التحليل...";

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "حدث خطأ في الخادم");
        }

        const data = await response.json();

        // بناء نص النتيجة
        let output = `نوع الإدخال: ${data.type}\n\n`;

        if (data.type === "text") {
            output += `التصنيف النهائي: ${data.result.final_label}\n`;
            output += `الثقة: ${data.result.final_confidence}\n\n`;

            data.result.details.forEach((detail, i) => {
                output += `المصدر ${i + 1}:\n`;
                output += `  التصنيف: ${detail.label}\n`;
                output += `  الثقة: ${detail.confidence}\n`;
                if (detail.url) output += `  الرابط: ${detail.url}\n`;
                output += "\n";
            });
        } else if (data.type === "url") {
            output += JSON.stringify(data.result, null, 2);
        }

        document.getElementById("result").textContent = output;

    } catch (err) {
        document.getElementById("result").textContent = `❌ خطأ: ${err.message}`;
    }
});
