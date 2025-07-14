// Глобальные переменные для управления сообщениями
let lastMessages = [];
let messageQueue = [];

// Расширенный список системных сообщений
const systemMessages = [
    { text: "> SYSTEM CHECK: <span class='success'>OK</span>", priority: 1 },
    { text: "> MEMORY ALLOCATION: <span class='success'>STABLE</span>", priority: 1 },
    { text: "> ENCRYPTION: <span class='success'>ACTIVE</span>", priority: 2 },
    { text: "> NETWORK THROUGHPUT: <span class='warning'>1.4 MB/S</span>", priority: 2 },
    { text: "> SCAN COMPLETE: <span class='success'>NO THREATS DETECTED</span>", priority: 1 },
    { text: "> DATABASE INDEXING: <span class='warning'>87% COMPLETE</span>", priority: 2 },
    { text: "> AI CORE: <span class='success'>ONLINE</span>", priority: 3 },
    { text: "> THERMAL SENSORS: <span class='warning'>42°C</span>", priority: 2 },
    { text: "> SECURITY PROTOCOLS: <span class='success'>ENGAGED</span>", priority: 3 },
    { text: "> DATA STREAM: <span class='success'>OPTIMAL</span>", priority: 1 },
    { text: "> FIREWALL: <span class='critical'>UNAUTHORIZED ACCESS ATTEMPT</span>", priority: 3 },
    { text: "> BACKUP SYSTEMS: <span class='success'>ONLINE</span>", priority: 2 },
    { text: "> POWER CONSUMPTION: <span class='warning'>78% CAPACITY</span>", priority: 2 },
    { text: "> NEURAL NETWORK: <span class='success'>SYNCHRONIZED</span>", priority: 3 },
    { text: "> VIDEO ANALYSIS: <span class='success'>PROCESSING</span>", priority: 2 },
    { text: "> DATA PACKETS: <span class='success'>127 RECEIVED</span>", priority: 1 },
    { text: "> ENCRYPTION KEYS: <span class='warning'>ROTATING</span>", priority: 3 },
    { text: "> SYSTEM INTEGRITY: <span class='critical'>CHECK REQUIRED</span>", priority: 3 }
];

// Функция для получения случайного сообщения без повторов
function getRandomMessage() {
    if (messageQueue.length === 0) {
        // Фильтруем сообщения, которые не показывались в последних 5
        const availableMessages = systemMessages.filter(msg =>
            !lastMessages.includes(msg.text)
        );

        // Если все сообщения были показаны, сбрасываем историю
        if (availableMessages.length === 0) {
            lastMessages = [];
            return systemMessages[Math.floor(Math.random() * systemMessages.length)];
        }

        // Сортируем по приоритету (более высокий приоритет имеет больше шансов)
        availableMessages.sort((a, b) => b.priority - a.priority);

        // Создаем очередь с учетом приоритетов
        availableMessages.forEach(msg => {
            const count = msg.priority * 2;
            for (let i = 0; i < count; i++) {
                messageQueue.push(msg);
            }
        });

        // Перемешиваем очередь
        messageQueue = messageQueue.sort(() => Math.random() - 0.5);
    }

    const nextMessage = messageQueue.pop();
    lastMessages.push(nextMessage.text);

    // Держим только последние 5 сообщений в истории
    if (lastMessages.length > 5) {
        lastMessages.shift();
    }

    return nextMessage;
}

async function updateStatus() {
    try {
        const response = await fetch('/buffer_status');
        const data = await response.json();
        document.getElementById('bufferStatus').textContent = data.buffer_size;
        document.getElementById('delay').textContent = Math.round(data.delay_seconds);

        // Обновляем системное время
        const now = new Date();
        const timeStr = now.toLocaleTimeString();
        document.getElementById('systemTime').textContent = timeStr;

    } catch (e) {
        console.error('Status update failed:', e);
    }
    setTimeout(updateStatus, 30000);
}

async function updateTortillaStats() {
    try {
        const response = await fetch('/tortilla_stats');
        const data = await response.json();



        const statsContainer = document.getElementById('tortillaStats');
        statsContainer.innerHTML = '';

        // Добавляем заголовок
        const header = document.createElement('div');
        header.className = 'analysis-item';
        header.innerHTML = `> ANALYSIS TIMESTAMP: <span class="success">${new Date().toLocaleTimeString()}</span>`;
        statsContainer.appendChild(header);

        // Добавляем полученные данные
        for (const [key, value] of Object.entries(data)) {
            const item = document.createElement('div');
            item.className = 'analysis-item';

            let displayValue = value;
            let valueClass = '';

            // Специальное форматирование для некоторых значений
            if (key.toLowerCase().includes('error') && value > 0) {
                valueClass = 'critical';
            } else if (key.toLowerCase().includes('warning')) {
                valueClass = 'warning';
            } else if (key.toLowerCase().includes('accuracy') || key.toLowerCase().includes('confidence')) {
                displayValue = `${(value * 100).toFixed(1)}%`;
                valueClass = value > 0.7 ? 'success' : value > 0.4 ? 'warning' : 'critical';
            }

            item.innerHTML = `> ${key.toUpperCase()}: <span class="${valueClass}">${displayValue}</span>`;
            statsContainer.appendChild(item);
        }




    } catch (e) {
        console.error('Failed to get tortilla stats:', e);

        const statsContainer = document.getElementById('tortillaStats');
        const errorItem = document.createElement('div');
        errorItem.className = 'analysis-item critical';
        errorItem.textContent = "> ERROR: ANALYSIS MODULE OFFLINE";
        statsContainer.appendChild(errorItem);
    }

    setTimeout(updateTortillaStats, 1000);
}

// Автопереподключение видео при ошибке
const video = document.getElementById('videoFeed');
video.onerror = function () {
    console.log('Reconnecting video...');
    video.src = '/video_feed?t=' + Date.now();
};

// Эффект печатающегося текста
function typeWriter(element, text, speed) {
    let i = 0;
    element.innerHTML = "";
    function typing() {
        if (i < text.length) {
            element.innerHTML += text.charAt(i);
            i++;
            setTimeout(typing, speed);
        }
    }
    typing();
}

// Функция для добавления системного сообщения
function addSystemMessage() {
    const message = getRandomMessage();
    const newElement = document.createElement('p');
    newElement.innerHTML = message.text;
    newElement.style.animation = "flicker 0.5s";

    const statusContainer = document.querySelector('.status');
    statusContainer.appendChild(newElement);

    // Удаляем старые сообщения
    const allMessages = document.querySelectorAll('.status p');
    if (allMessages.length > 8) {
        allMessages[3].remove(); // Удаляем сообщение после статических элементов
    }
}

// При загрузке страницы
window.onload = function () {
    updateStatus();
    updateTortillaStats();

    // Эффект печатающегося заголовка
    const titleElement = document.querySelector('h2');
    const originalTitle = titleElement.textContent;
    titleElement.textContent = "";
    typeWriter(titleElement, originalTitle, 100);

    // Запускаем показ системных сообщений
    setInterval(addSystemMessage, 2500);

    // Первые несколько сообщений
    setTimeout(addSystemMessage, 500);
    setTimeout(addSystemMessage, 1500);
    setTimeout(addSystemMessage, 2500);
};
