import os
import openai

# Initialize empty lists to store time and notifications
time_list = []
notification_list = []

# Function to add a notification
def add_notification(time, notification):
    time_list.append(time)
    notification_list.append(notification)

# Function to view all notifications
def view_notifications():
    hours = {}
    for i in range(len(time_list)):
        hour = int(time_list[i][:2])
        if hour not in hours:
            hours[hour] = []
        hours[hour].append((time_list[i], notification_list[i]))
    
    for hour in sorted(hours.keys()):
        print(f"Hour: {hour}:00")
        for time, notification in hours[hour]:
            print(f"  Time: {time} | Notification: {notification}")

# Function to summarize notifications
def summarize_notifications(indexes):
    summary = {}
    for i in indexes:
        hour = int(time_list[i][:2])
        if hour not in summary:
            summary[hour] = []
        summary[hour].append(notification_list[i])
    
    for hour in sorted(summary.keys()):
        print(f"Hour: {hour}:00")
        print(f"  Notifications: {', '.join(summary[hour])}")

# Function to create a chat completion
def create_chat_completion():
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )
    response = client.chat.completions.create(
        model='Meta-Llama-3.1-8B-Instruct',
        messages=[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"User A commented on your post. User B sent you a friend request. You received a new message from User C."}],
        temperature =  0.1,
        top_p = 0.1
    )
    return response

# Function to process notification
def process_notification(index):
    notification = notification_list[index]
    summary = {}
    for i in range(len(time_list)):
        if notification in notification_list[i]:
            hour = int(time_list[i][:2])
            if hour not in summary:
                summary[hour] = []
            summary[hour].append(notification_list[i])
    
    for hour in sorted(summary.keys()):
        print(f"Hour: {hour}:00")
        print(f"  Notifications: {', '.join(summary[hour])}")

# Function to send data to API
def send_data_to_api(data):
    client = openai.OpenAI(
        api_key="bb15ea8b-d005-4696-9ee3-5dc8ba66df6e",
        base_url="https://api.sambanova.ai/v1",
    )
    data = data + " summarise the above message"
    response = client.chat.completions.create(
        model='Meta-Llama-3.1-8B-Instruct',
        messages=[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":data}],
        temperature =  0.1,
        top_p = 0.1
    )
    return response

# Function to summarize all notifications
def summarize_all_notifications():
    summary = {}
    for i in range(len(time_list)):
        hour = int(time_list[i][:2])  # Extract the hour from the time
        if hour not in summary:
            summary[hour] = []
        summary[hour].append(notification_list[i])  # Group notifications by hour
    
    # Print the summarized notifications
    for hour in sorted(summary.keys()):
        print(f"Hour: {hour}:00")
        print(f"  Notifications: {', '.join(summary[hour])}")
    return summary

# Function to create a final summary of the day
def create_final_summary():
    total_notifications = len(notification_list)
    interactions = {}
    for notification in notification_list:
        action = notification.split()[1]  # Extract the action (liked, shared, etc.)
        interactions[action] = interactions.get(action, 0) + 1

    # Create a summary string
    interaction_summary = ", ".join([f"{count} {action}(s)" for action, count in interactions.items()])
    return f"Today, you had {total_notifications} notifications on your social media, including {interaction_summary}. It seems like your day was quite engaging!"

# Example usage
add_notification("08:15", "Alice liked your photo.")
add_notification("09:00", "Bob shared your post.")
add_notification("10:20", "Charlie commented on your post.")
add_notification("10:45", "Diana sent you a friend request.")
add_notification("11:30", "Eve started following you.")
add_notification("13:00", "Frank sent you a new message.")
add_notification("15:45", "Grace mentioned you in a comment.")
add_notification("17:20", "Henry invited you to an event.")
add_notification("19:05", "Ivy tagged you in a photo.")

# Summarize all notifications
print("\nSummary of All Notifications:")
hourly_summary = summarize_all_notifications()

# Create and print the final summary
print("\nFinal Summary of Your Day:")
final_summary = create_final_summary()
print(final_summary)


