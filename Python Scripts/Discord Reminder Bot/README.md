# Discord Reminder Bot

A simple, customizable Discord bot that sends scheduled reminders to specified channels.

## System Requirements

- Python 3.8 or higher
- A Discord account with permissions to create bots
- A server where you have permission to add bots

## Setup Instructions

### Step 1: Create a Discord Application and Bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" and give it a name (e.g., "Reminder Bot")
3. Go to the "Bot" tab and click "Add Bot"
4. Under the TOKEN section, click "Copy" to copy your bot token (keep this secure!)
5. Under "Privileged Gateway Intents", enable "Message Content Intent" if available

### Step 2: Invite the Bot to Your Server

1. In the Developer Portal, go to OAuth2 > URL Generator
2. Select the "bot" scope
3. Select these permissions:
   - Send Messages
   - View Channels
   - Read Message History
4. Copy the generated URL and open it in your browser
5. Select your server and authorize the bot

### Step 3: Get Your Discord ID and Channel ID

1. In Discord Settings > Advanced, enable "Developer Mode"
2. To get your Discord ID: Right-click on your username and select "Copy ID"
3. To get the Channel ID: Right-click on the channel you want to send reminders to and select "Copy ID"

### Step 4: Install Dependencies

1. Create a folder for your bot
2. Copy the `discord_reminder.py` script and `requirements.txt` into this folder
3. Open a terminal or command prompt in this folder
4. Run: `pip install -r requirements.txt`

### Step 5: Configure the Bot

1. Create a file named `.env` in the same folder with the following content:
```
DISCORD_TOKEN=your_bot_token_here
CHANNEL_ID=your_channel_id_here
```

2. In the `discord_reminder.py` file, update your Discord ID:
```python
myid = "<@your_discord_id_here>"
```

3. Configure your reminders in the REMINDERS section. Each reminder follows this format:
```python
[day_of_week, hour, minute, f"emoji **TITLE**: {myid} Your message"]
```
   - `day_of_week`: 0-6 (Monday to Sunday)
   - `hour`: 0-23 (24-hour format)
   - `minute`: 0-59

### Step 6: Run the Bot

1. In your terminal or command prompt, run:
```
python discord_reminder.py
```

2. You should see a message that the bot has logged in
3. The bot will now send reminders according to your configured schedule

## Running the Bot in the Background

### Windows
Create a batch file (start_bot.bat) with:
```
@echo off
start pythonw discord_reminder.py
```

### macOS/Linux
```
nohup python discord_reminder.py > bot.log 2>&1 &
```

## Setting Up as a Service (for 24/7 operation)

### Linux (systemd)
1. Create a service file:
```
sudo nano /etc/systemd/system/discord-reminder.service
```

2. Add the following content:
```
[Unit]
Description=Discord Reminder Bot
After=network.target

[Service]
ExecStart=/usr/bin/python3 /path/to/discord_reminder.py
WorkingDirectory=/path/to/bot/folder
StandardOutput=append:/path/to/bot/folder/bot.log
StandardError=append:/path/to/bot/folder/bot.log
Restart=always
User=your_username

[Install]
WantedBy=multi-user.target
```

3. Enable and start the service:
```
sudo systemctl enable discord-reminder.service
sudo systemctl start discord-reminder.service
```

## Troubleshooting

- **Bot not responding**: Make sure your bot token is correct and the bot has the necessary permissions
- **No reminders being sent**: Verify the channel ID is correct and the bot has permissions to send messages in that channel
- **Wrong reminder times**: Check that your system clock is correctly set and synchronized

## Example Reminders

The script includes examples for each day of the week. Uncomment or modify them as needed:

- Monday: Team meeting reminder at 9:00 AM
- Tuesday: Progress report reminder at 10:30 AM
- Wednesday: Mid-week check-in at 12:00 PM
- Thursday: Project update due reminder at 11:00 AM
- Friday: Weekly metrics reminder at 9:30 AM
- Saturday: Weekend tasks reminder at 10:00 AM
- Sunday: Timesheet submission reminder at 3:00 PM