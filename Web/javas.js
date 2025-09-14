document.addEventListener("DOMContentLoaded", () => {
    const body = document.body;
    const toggleBtn = document.getElementById("toggle-mode");
    const modeIcon = document.getElementById("mode-icon");
    const sendBtn = document.getElementById("send-btn");
    const sendIcon = document.querySelector(".SendIcon");
    const chatBox = document.getElementById("chat-box");
    const input = document.getElementById("user-input");

    // Lấy chế độ đã lưu
    const savedTheme = localStorage.getItem("theme");

    if (savedTheme === "dark") {
        body.classList.add("dark-mode");
        modeIcon.src = "/Web/images/BongDenWhite.png";
        sendIcon.src = "/Web/images/MuiTenWhite.png";
    } else {
        body.classList.add("light-mode"); // mặc định sáng
        modeIcon.src = "/Web/images/BongDenBlack.png";
        sendIcon.src = "/Web/images/MuiTenBlack.png";
    }

    // Lời chào khi mở chat
    chatBox.innerHTML = `<div class="WelcomeMessage" id="welcome-message">Chào bạn, bạn có thắc mắc gì?</div>`;

    // Sự kiện toggle chế độ sáng/tối
    toggleBtn.addEventListener("click", () => {
        if (body.classList.contains("dark-mode")) {
            body.classList.remove("dark-mode");
            body.classList.add("light-mode");
            localStorage.setItem("theme", "light");
            modeIcon.src = "/Web/images/BongDenBlack.png";
            sendIcon.src = "/Web/images/MuiTenBlack.png";
        } else {
            body.classList.remove("light-mode");
            body.classList.add("dark-mode");
            localStorage.setItem("theme", "dark");
            modeIcon.src = "/Web/images/BongDenWhite.png";
            sendIcon.src = "/Web/images/MuiTenWhite.png";
        }
    });

    // Hàm gửi tin nhắn
    async function sendMessage() {
        const userText = input.value.trim();
        if (!userText) return;

        // Xóa lời chào nếu còn
        const welcomeMsg = document.getElementById("welcome-message");
        if (welcomeMsg) welcomeMsg.remove();

        // Hiển thị tin nhắn người dùng
        chatBox.innerHTML += `<div class="UserMessage">${userText}</div>`;
        input.value = "";

        try {
            // Gửi API
            const res = await fetch("http://localhost:5000/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: userText })
            });
            const data = await res.json();

            // Hiển thị phản hồi từ bot
            chatBox.innerHTML += `<div class="BotMessage">${data.answer}</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
        } catch (e) {
            chatBox.innerHTML += `<div class="BotMessage" style="color:red;">Lỗi kết nối API!</div>`;
        }
    }

    // Click nút gửi
    sendBtn.addEventListener("click", sendMessage);

    // Enter để gửi
    input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            sendMessage();
        }
    });
});
