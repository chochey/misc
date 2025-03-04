import discord
from discord.ext import tasks, commands
import datetime
import os
import json
import re
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
current_dir = os.path.dirname(os.path.abspath(__file__))
REMINDERS_FILE = os.path.join(current_dir, "reminders.json")
logger.info(f"Reminders will be saved to: {REMINDERS_FILE}")

# Default reminders
DEFAULT_REMINDERS = [
    # ========== SATURDAY REMINDERS (day 5) ==========
    [5, 23, 59, f"**REMINDER**: {myid} Dominos email analysis."],
    # ========== SUNDAY REMINDERS (day 6) ==========
    [6, 15, 30, f"**REMINDER**: {myid} Submit timesheet for Arrow."],
]

# Day name mapping
DAY_MAPPING = {
    "monday": 0, "mon": 0, "m": 0,
    "tuesday": 1, "tue": 1, "tu": 1, "t": 1,
    "wednesday": 2, "wed": 2, "w": 2,
    "thursday": 3, "thu": 3, "th": 3,
    "friday": 4, "fri": 4, "f": 4,
    "saturday": 5, "sat": 5, "sa": 5,
    "sunday": 6, "sun": 6, "su": 6,
    "today": datetime.datetime.now().weekday(),
    "tomorrow": (datetime.datetime.now().weekday() + 1) % 7
}


# Load reminders from file or use defaults
def load_reminders():
    try:
        if os.path.exists(REMINDERS_FILE):
            logger.info(f"Loading reminders from {REMINDERS_FILE}")
            with open(REMINDERS_FILE, 'r') as f:
                reminders = json.load(f)
                logger.info(f"Loaded {len(reminders)} reminders")
                return reminders
        else:
            logger.info(f"Reminders file not found at {REMINDERS_FILE}. Using default reminders.")
            # Save default reminders immediately to create the file
            save_reminders(DEFAULT_REMINDERS)
            return DEFAULT_REMINDERS
    except Exception as e:
        logger.error(f"Error loading reminders: {e}")
        return DEFAULT_REMINDERS


# Save reminders to file
def save_reminders(reminders):
    try:
        logger.info(f"Saving {len(reminders)} reminders to {REMINDERS_FILE}")
        with open(REMINDERS_FILE, 'w') as f:
            json.dump(reminders, f, indent=2)
        logger.info("Reminders saved successfully")
    except Exception as e:
        logger.error(f"Error saving reminders: {e}")


# Parse time string in various formats
def parse_time(time_str):
    time_str = time_str.lower().strip()

    # Check for 24-hour format (13:45)
    match24 = re.match(r'^(\d{1,2}):(\d{2})$', time_str)
    if match24:
        hour = int(match24.group(1))
        minute = int(match24.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute

    # Check for 12-hour format with am/pm (1:45pm)
    match12 = re.match(r'^(\d{1,2}):(\d{2})\s*(am|pm)$', time_str)
    if match12:
        hour = int(match12.group(1))
        minute = int(match12.group(2))
        am_pm = match12.group(3)

        if 1 <= hour <= 12 and 0 <= minute <= 59:
            if am_pm == "pm" and hour < 12:
                hour += 12
            elif am_pm == "am" and hour == 12:
                hour = 0
            return hour, minute

    # Check for shorthand format (1pm)
    match_short = re.match(r'^(\d{1,2})\s*(am|pm)$', time_str)
    if match_short:
        hour = int(match_short.group(1))
        am_pm = match_short.group(2)

        if 1 <= hour <= 12:
            if am_pm == "pm" and hour < 12:
                hour += 12
            elif am_pm == "am" and hour == 12:
                hour = 0
            return hour, 0

    # Check for just hour format (13)
    if time_str.isdigit():
        hour = int(time_str)
        if 0 <= hour <= 23:
            return hour, 0

    # Check for "now"
    if time_str == "now":
        now = datetime.datetime.now()
        return now.hour, now.minute

    # If all parsing fails, return None
    return None


# Format time for display
def format_time_display(hour, minute):
    if hour < 12:
        period = "AM"
        display_hour = hour if hour > 0 else 12
    else:
        period = "PM"
        display_hour = hour - 12 if hour > 12 else 12

    return f"{display_hour}:{minute:02d} {period}"


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

    logger.debug(f"Checking reminders at {current_day} day, {current_hour}:{current_minute}")

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
async def add_reminder(ctx, day_str, time_str, *, message):
    """
    Add a new reminder with flexible day and time format
    Examples:
      !addreminder mon 3:00pm Team meeting
      !addreminder tomorrow 14:30 Call with client
      !addreminder friday 5pm Weekend planning
      !addreminder today now Quick reminder
    """
    # Parse day
    day_str = day_str.lower()
    day = None

    # Try to parse as a number first
    if day_str.isdigit():
        day_num = int(day_str)
        if 0 <= day_num <= 6:
            day = day_num

    # Try to parse as a day name
    if day is None and day_str in DAY_MAPPING:
        day = DAY_MAPPING[day_str]

    if day is None:
        await ctx.send("❌ Invalid day format. Please use a day number (0-6), name (monday), or shorthand (mon).")
        return

    # Parse time
    time_result = parse_time(time_str)
    if time_result is None:
        await ctx.send("❌ Invalid time format. Please use 24-hour (14:30) or 12-hour (2:30pm) format.")
        return

    hour, minute = time_result

    # Format the message with the user ID and time
    formatted_time = format_time_display(hour, minute)
    formatted_message = f"**REMINDER**: {myid} {message}"

    # Add the reminder
    REMINDERS.append([day, hour, minute, formatted_message])
    save_reminders(REMINDERS)

    # Convert day number to day name
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_name = day_names[day]

    await ctx.send(f"✅ Reminder added for {day_name} at {formatted_time}: {message}")


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

        # Format time for display
        formatted_time = format_time_display(hour, minute)

        reminders_by_day[day_name].append((i, formatted_time, message))

    # Build the response
    response = "📅 **Current Reminders:**\n\n"

    for day in day_names:
        if day in reminders_by_day:
            response += f"**{day}:**\n"
            for i, time, message in reminders_by_day[day]:
                response += f"  `{i}`: {time} - {message}\n"
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
    formatted_time = format_time_display(hour, minute)

    del REMINDERS[index]
    save_reminders(REMINDERS)

    await ctx.send(f"🗑️ Removed reminder: {day_name} at {formatted_time} - {message}")


@bot.command(name="help_reminders")
async def help_reminders(ctx):
    """Show help for reminder commands"""
    help_text = """
📝 **Reminder Bot Commands**

`!addreminder <day> <time> <message>`
  Add a new reminder with flexible formats
  - day: 0-6, monday, mon, today, tomorrow
  - time: 14:30, 2:30pm, 2pm, now
  - message: Your reminder message

`!listreminders`
  Show all current reminders with their IDs

`!removereminder <id>`
  Remove a reminder by its ID number

**Examples:**
  `!addreminder monday 3pm Team meeting`
  Adds a reminder for Monday at 3:00 PM

  `!addreminder today now Urgent reminder`
  Adds a reminder for right now

  `!addreminder tomorrow 9:30am Morning check-in`
  Adds a reminder for tomorrow morning
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