import discord
from discord.ext import tasks, commands
import datetime
import os
import json
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('discord_reminder')

# Load environment variables from .env file
load_dotenv()

# Discord bot setup with command support
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent for commands
bot = commands.Bot(command_prefix='!', intents=intents)

# Configuration
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
CHANNEL_ID = int(os.getenv('CHANNEL_ID'))  # The channel ID to send reminders to
myid = "<@142171299844718592>"

# File to store reminders
REMINDERS_FILE = "reminders.json"

# Default reminders
DEFAULT_REMINDERS = [
    # ========== SATURDAY REMINDERS (day 5) ==========
    [5, 23, 59, f"**REMINDER**: {myid} Dominos email analysis."],
    # ========== SUNDAY REMINDERS (day 6) ==========
    [6, 15, 30, f"🔮 **REMINDER**: {myid} Submit timesheet for Arrow."],
]


# Load reminders from file or use defaults
def load_reminders():
    try:
        if os.path.exists(REMINDERS_FILE):
            with open(REMINDERS_FILE, 'r') as f:
                return json.load(f)
        else:
            return DEFAULT_REMINDERS
    except Exception as e:
        logger.error(f"Error loading reminders: {e}")
        return DEFAULT_REMINDERS


# Save reminders to file
def save_reminders(reminders):
    try:
        with open(REMINDERS_FILE, 'w') as f:
            json.dump(reminders, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving reminders: {e}")


# Initialize reminders
REMINDERS = load_reminders()


@bot.event
async def on_ready():
    """Event triggered when the bot is ready and connected to Discord."""
    logger.info(f'Logged in as {bot.user.name} ({bot.user.id})')
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
                channel = bot.get_channel(CHANNEL_ID)
                if channel:
                    await channel.send(message)
                    logger.info(f"Sent reminder: {message}")
                else:
                    logger.error(f"Could not find channel with ID {CHANNEL_ID}")
            except Exception as e:
                logger.error(f"Error sending reminder: {e}")


@bot.command(name="addreminder")
async def add_reminder(ctx, day: int, hour: int, minute: int, *, message: str):
    """
    Add a new reminder
    Usage: !addreminder <day> <hour> <minute> <message>
    day: 0-6 (Monday-Sunday)
    hour: 0-23
    minute: 0-59
    message: The reminder message
    """
    if not 0 <= day <= 6:
        await ctx.send("❌ Day must be between 0 (Monday) and 6 (Sunday)")
        return

    if not 0 <= hour <= 23:
        await ctx.send("❌ Hour must be between 0 and 23")
        return

    if not 0 <= minute <= 59:
        await ctx.send("❌ Minute must be between 0 and 59")
        return

    REMINDERS.append([day, hour, minute, message])
    save_reminders(REMINDERS)

    # Convert day number to day name
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_name = day_names[day]

    await ctx.send(f"✅ Reminder added for {day_name} at {hour:02d}:{minute:02d}: {message}")


@bot.command(name="listreminders")
async def list_reminders(ctx):
    """List all current reminders"""
    if not REMINDERS:
        await ctx.send("No reminders set.")
        return

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Group reminders by day
    reminders_by_day = {}
    for i, (day, hour, minute, message) in enumerate(REMINDERS):
        day_name = day_names[day]
        if day_name not in reminders_by_day:
            reminders_by_day[day_name] = []
        reminders_by_day[day_name].append((i, hour, minute, message))

    # Build the response
    response = "📅 **Current Reminders:**\n\n"

    for day in day_names:
        if day in reminders_by_day:
            response += f"**{day}:**\n"
            for i, hour, minute, message in reminders_by_day[day]:
                response += f"  `{i}`: {hour:02d}:{minute:02d} - {message}\n"
            response += "\n"

    await ctx.send(response)


@bot.command(name="removereminder")
async def remove_reminder(ctx, index: int):
    """
    Remove a reminder by its index
    Usage: !removereminder <index>
    Use !listreminders to see indices
    """
    if index < 0 or index >= len(REMINDERS):
        await ctx.send(f"❌ Invalid index. Please use a number between 0 and {len(REMINDERS) - 1}")
        return

    day, hour, minute, message = REMINDERS[index]
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_name = day_names[day]

    del REMINDERS[index]
    save_reminders(REMINDERS)

    await ctx.send(f"🗑️ Removed reminder: {day_name} at {hour:02d}:{minute:02d} - {message}")


@bot.command(name="help_reminders")
async def help_reminders(ctx):
    """Show help for reminder commands"""
    help_text = """
📝 **Reminder Bot Commands**

`!addreminder <day> <hour> <minute> <message>`
  Add a new reminder
  - day: 0 (Monday) to 6 (Sunday)
  - hour: 0-23 (24-hour format)
  - minute: 0-59
  - message: Your reminder message

`!listreminders`
  Show all current reminders with their IDs

`!removereminder <id>`
  Remove a reminder by its ID number

**Examples:**
  `!addreminder 1 15 0 📞 Call with client`
  Adds a reminder for Tuesday at 3:00 PM

  `!removereminder 0`
  Removes the reminder with ID 0
"""
    await ctx.send(help_text)


def main():
    """Main function to start the bot."""
    if not DISCORD_TOKEN:
        logger.error("No Discord token found. Please set the DISCORD_TOKEN environment variable.")
        return

    logger.info("Starting Discord reminder bot...")
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()