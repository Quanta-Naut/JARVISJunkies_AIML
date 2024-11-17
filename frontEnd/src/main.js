const openBtn = document.getElementById('openBtn');
const closeBtn = document.getElementById('closeBtn');
const popupWindow = document.getElementById('popupWindow');
const messagesContainer = document.getElementById('messagesContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const ctx = document.getElementById('myChart');


new Chart(ctx, {
  type: 'line',
  data: {
    labels: ['Facebook', 'Instagram', 'Twitter', 'Whatsapp', 'YouTube', 'LinkedIn'],
    datasets: [{
      data: [12, 19, 3, 5, 2, 3],
      borderWidth: 1,
      borderColor: 'rgba(75, 192, 192, 1)', // Optional: Add a border color for better visuals
      backgroundColor: 'rgba(75, 192, 192, 0.2)', // Optional: Add a fill color
      tension: 0.4 // This makes the line smooth
    }]
  },
  options: {
    plugins: {
      legend: {
        display: false // Disable legend
      }
    },
    scales: {
      x: {
        ticks: {
          font: {
            size: 15// Increase x-axis label font size
          }
        }
      },
      y: {
        beginAtZero: true,
        ticks: {
          font: {
            size: 15 // Increase y-axis label font size
          }
        }
      }
    }
  }
});