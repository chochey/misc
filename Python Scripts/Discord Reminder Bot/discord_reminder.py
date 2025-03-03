import discord
from discord.ext import tasks
import datetime
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('discord_reminder')

# Load environment variables from .env file
load_dotenv()

# Discord bot setup
intents = discord.Intents.default()
client = discord.Client(intents=intents)

# Configuration
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
CHANNEL_ID = int(os.getenv('CHANNEL_ID'))  # The channel ID to send reminders to
myid = "<@142171299844718592>"

# Configuration
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
CHANNEL_ID = int(os.getenv('CHANNEL_ID'))  # The channel ID to send reminders to

# Reminder configuration
# Format: [day_of_week, hour, minute, message]
# day_of_week: 0-6 (Monday-Sunday)
REMINDERS = [
    # ========== MONDAY REMINDERS (day 0) ==========
    #[0, 14, 43, f"📋 **REMINDER**: {myid} Set goals for the week."],

    # ========== TUESDAY REMINDERS (day 1) ==========
    # [1, 15, 0, f"📞 **REMINDER**: {myid} Client call at 3:30 PM."],

    # ========== WEDNESDAY REMINDERS (day 2) ==========
    # [2, 14, 30, f"🤝 **REMINDER**: {myid} Department meeting at 3 PM."],

    # ========== THURSDAY REMINDERS (day 3) ==========
    # [3, 16, 0, f"🔄 **REMINDER**: {myid} Review this week's progress."],

    # ========== FRIDAY REMINDERS (day 4) ==========
    # [4, 15, 0, f"🎉 **REMINDER**: {myid} Wrap up tasks for the weekend!"],

    # ========== SATURDAY REMINDERS (day 5) ==========
    # [5, 14, 0, f"🛒 **REMINDER**: {myid} Grocery shopping reminder."],

    # ========== SUNDAY REMINDERS (day 6) ==========
    # [6, 18, 0, f"🔮 **REMINDER**: {myid} Prepare for the upcoming week."],
      [6, 15, 30, f"🔮 **REMINDER**: {myid} Submit timesheet for Arrow."],

    # You can uncomment any of the commented reminders above to activate them
    # Or add your own reminders following the same format
]

# Add more reminders here, for example:
# [0, 9, 0, "🗓️ **REMINDER**: Weekly team meeting at 10:00 AM"],
# [4, 16, 30, "💼 **REMINDER**: Submit weekly report before end of day"],

@client.event
async def on_ready():
    """Event triggered when the bot is ready and connected to Discord."""
    logger.info(f'Logged in as {client.user.name} ({client.user.id})')
    check_reminders.start()


@tasks.loop(minutes=1)
async def check_reminders():
    """Check if any reminders need to be sent at the current time."""
    now = datetime.datetime.now()
    current_day = now.weekday()  # 0-6 (Monday-Sunday)
    current_hour = now.hour
    current_minute = now.minute

    for day, hour, minute, message in REMINDERS:
        if current_day == day and current_hour == hour and current_minute == minute:
            try:
                channel = client.get_channel(CHANNEL_ID)
                if channel:
                    await channel.send(message)
                    logger.info(f"Sent reminder: {message}")
                else:
                    logger.error(f"Could not find channel with ID {CHANNEL_ID}")
            except Exception as e:
                logger.error(f"Error sending reminder: {e}")


def main():
    """Main function to start the bot."""
    if not DISCORD_TOKEN:
        logger.error("No Discord token found. Please set the DISCORD_TOKEN environment variable.")
        return

    logger.info("Starting Discord reminder bot...")
    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()