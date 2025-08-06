import time
import tiktoken
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.console import Group
from dataclasses import dataclass
from typing import List, Literal

MODEL_NAME = "gpt-oss:20b"
ENCODING_NAME = "o200k_base"

console = Console()

client = OpenAI(
    base_url="http://localhost:11434/v1",  # Local Ollama API
    api_key="ollama"                       # Dummy key
)
 
# Try basic tiktoken encoding
try:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
except Exception as e:
    console.print(f"[red]Warning: Could not load encoding: {e}[/red]")
    console.print(f"[yellow] Defaulting to chunk counting [/yellow]")
    # Fallback to chunk counting
    encoding = None


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


def main(messages: List[Message]):
    start_time = time.time()
    first_token_time = None
    token_count = 0
    response_text = ""
    streaming_complete = False
    
    console.print(Panel("🤖 Local LLM Chat", style="bold blue"))

    # Convert dataclass messages to dict format for OpenAI API
    api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=api_messages,
        stream=True
    )
    
    # Create a live display with both metrics and streaming text
    def generate_display():
        current_time = time.time()
        elapsed_time = current_time - start_time
        running_tps = token_count / elapsed_time if elapsed_time > 0 else 0
        
        # Create the metrics table
        table = Table(title="📊 Live Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Tokens Generated", str(token_count))
        table.add_row("Elapsed Time", f"{elapsed_time:.2f}s")
        table.add_row("Running TPS", f"{running_tps:.2f}")
        
        if first_token_time:
            time_to_first = first_token_time - start_time
            table.add_row("Time to First Token", f"{time_to_first:.2f}s")
        
        metrics_panel = Panel(table, style="green")
        
        # Show only last 10 lines while streaming, full text when complete
        if streaming_complete:
            display_text = response_text
        else:
            lines = response_text.split('\n')
            if len(lines) > 10:
                display_text = "...\n" + '\n'.join(lines[-10:])
            else:
                display_text = response_text if response_text else "Generating..."
        
        # Create response text panel
        response_panel = Panel(
            display_text,
            title="🤖 Response",
            style="white",
            expand=True,
            height=None
        )
        
        # Group metrics and response together
        return Group(metrics_panel, response_panel)
    
    # Handle streaming response
    with Live(generate_display(), refresh_per_second=4, console=console, auto_refresh=True) as live:
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                response_text += content
                
                # Calculate actual tokens using tiktoken or fallback
                if encoding:
                    tokens_in_chunk = len(encoding.encode(content))
                else:
                    # Fallback to counting stream entries (chunks)
                    if content:
                        tokens_in_chunk = 1
                    else:
                        tokens_in_chunk = 0
                token_count += tokens_in_chunk
                
                # Record time to first token
                if first_token_time is None and token_count > 0:
                    first_token_time = time.time()
                
                # Update the live display
                live.update(generate_display())
    
    # Mark streaming as complete and show final display
    streaming_complete = True
    live.update(generate_display())
    
    # Calculate final metrics
    end_time = time.time()
    total_time = end_time - start_time
    final_tps = token_count / total_time if total_time > 0 else 0
    
    console.print(Panel(
        f"📊 Final Performance Metrics:\n"
        f"• Total tokens: {token_count}\n"
        f"• Total time: {total_time:.2f}s\n"
        f"• Final tokens per second: {final_tps:.2f}\n"
        f"• Time to first token: {first_token_time - start_time:.2f}s" if first_token_time else "• Time to first token: N/A",
        title="Final Stats",
        style="bold green"
    ))


if __name__ == "__main__":
    # Example usage with dataclass messages
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Is motorway good for selling my car?")
    ]
    main(messages)
