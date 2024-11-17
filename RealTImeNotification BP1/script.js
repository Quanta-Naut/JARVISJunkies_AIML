// Settings button functionality
document.getElementById("settings-btn").addEventListener("click", function () {
  const settingsMenu = document.getElementById("settings-menu");
  settingsMenu.style.display =
    settingsMenu.style.display === "block" ? "none" : "block";
});

// Close the settings menu if clicked outside
window.addEventListener("click", function (event) {
  const settingsMenu = document.getElementById("settings-menu");
  const hamburgerMenu = document.getElementById("menu");
  if (!event.target.closest(".settings-icon")) {
    settingsMenu.style.display = "none";
  }
  if (!event.target.closest("#hamburger")) {
    hamburgerMenu.style.display = "none";
  }
});

// Notification count logic
let notificationCount = 0;

// Store notifications with timestamps
let notifications = [];

// Function to format the timestamp
function formatTimestamp(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// Like button functionality
document.getElementById("like-button").addEventListener("click", function () {
  console.log("Like button clicked!"); // Debugging line
  notificationCount++;
  const timestamp = new Date();
  notifications.push(
    `Someone liked your post at ${formatTimestamp(timestamp)}`
  ); // Add notification with timestamp
  document.getElementById("notification-count").innerText = notificationCount;
  alert("Someone liked your post!"); // This should trigger the alert
});

// Comment button functionality
document
  .getElementById("comment-button")
  .addEventListener("click", function () {
    notificationCount++;
    const timestamp = new Date();
    notifications.push(
      `Someone commented on your post at ${formatTimestamp(timestamp)}`
    ); // Add notification with timestamp
    document.getElementById("notification-count").innerText = notificationCount;
    alert("Someone commented on your post!");
  });

// Bell icon functionality
document
  .getElementById("notification-icon")
  .addEventListener("click", function () {
    const modal = document.getElementById("notification-modal");
    const notificationList = document.getElementById("notification-list");

    // Clear previous notifications
    notificationList.innerHTML = "";

    // Add notifications to the list
    notifications.forEach(notification => {
      const li = document.createElement("li");
      li.textContent = notification;
      notificationList.appendChild(li);
    });

    modal.style.display = "block"; // Show the modal
  });

// Close button functionality
document.getElementById("close-modal").addEventListener("click", function () {
  document.getElementById("notification-modal").style.display = "none"; // Hide the modal
});

// Close modal when clicking outside of it
window.addEventListener("click", function (event) {
  const modal = document.getElementById("notification-modal");
  if (event.target === modal) {
    modal.style.display = "none"; // Hide the modal
  }
});
