from typing import Final
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv


# Get the path to the current directory
current_directory = os.getcwd()
# Specify the path to the .env file relative to the current directory
dotenv_path = os.path.join(current_directory, '.env')
# Load the environment variables from the .env file
load_dotenv(dotenv_path)

# Replace 'YOUR_BOT_API_TOKEN' with your actual bot API token
BOT_API_TOKEN = os.getenv('TELEGRAM_TOKEN')
BOT_USERNAME = '@yenju_Trade_bot'

# execute notebook
import asyncio
import threading
from nbformat import read, write
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
from nbconvert.preprocessors import CellExecutionError

def restart_kernal_runall(notebook_path):
    
    # Read the notebook
    with open(notebook_path, "r") as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)
    
    # Create an ExecutePreprocessor to run the cells
    executor = ExecutePreprocessor(timeout=None, kernel_name="python3")
    
    try:
        # Execute all the cells in the notebook
        executor.preprocess(notebook_content, {"metadata": {"path": "./"}})
    
        # Save the executed notebook
        with open(notebook_path, "w") as output_notebook:
            write(notebook_content, output_notebook)
    
        print("Notebook execution completed.")
    except CellExecutionError as e:
        print(f"Error executing cell {e.cell_name}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


async def start_long_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print('start long')
    long_path = "LONG_1h_4h_ema_WS.ipynb"

    # Use threading to run the notebook execution
    def execute_notebook():
        output = restart_kernal_runall(long_path)
        context.bot.send_message(chat_id=update.effective_chat.id, text=output)
    
    # Start the notebook execution in a separate thread
    thread = threading.Thread(target=execute_notebook)
    thread.start()

    await update.message.reply_text("Running long command in the background...")


async def start_short_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print('start short')
    short_path = "SHORT_1h_4h_ema_WS.ipynb"

        # Use threading to run the notebook execution
    def execute_notebook():
        output = restart_kernal_runall(short_path)
        context.bot.send_message(chat_id=update.effective_chat.id, text=output)
        
    # Start the notebook execution in a separate thread
    thread = threading.Thread(target=execute_notebook)
    thread.start()

    await update.message.reply_text("Running short command in the background...")


def handle_response(text: str) -> str:
    processed: str = text.lower()
    if 'hello' in processed:
        return 'hello...'
    
    return 'unknown...'


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        print("Received a text message:", update.message.text)
        # Rest of your code
    except Exception as e:
        print(f"Error in handle_message: {e}")

    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')

    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response: str = handle_response(new_text)
        else:
            return
    else:
        response: str = handle_response(text)
    
    print('Bot:', response)
    await update.message.reply_text(response)

    
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error: {context.error}')


if __name__ == '__main__':
    print('Starting bot...')
    app = Application.builder().token(BOT_API_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler('startlong', start_long_command))
    app.add_handler(CommandHandler('startshort', start_short_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Errors
    app.add_error_handler(error)

    print('Polling...')
    
    try:
        app.run_polling(poll_interval=3)
    except Exception as e:
        print(f"An error occurred: {e}")
