"""Command-line interface for Arabic OCR."""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
import json

app = typer.Typer(help="Arabic OCR System CLI")
console = Console()

# Commands will be added here

@app.command()
def predict(
    image: Path = typer.Argument(..., help="Path to image file"),
    model: Path = typer.Option("models/best_model.pth", help="Path to model checkpoint"),
    output: Path = typer.Option(None, help="Output file for results"),
    beam_width: int = typer.Option(5, help="Beam width for decoding"),
    show_confidence: bool = typer.Option(True, help="Show confidence score"),
):
    """
    Predict text from a single image.
    """
    from src.inference import ArabicOCRPredictor

    console.print(f"[bold blue]Loading model from {model}...[/bold blue]")

    try:
        predictor = ArabicOCRPredictor(model_path=str(model), beam_width=beam_width)
    except Exception as e:
        console.print(f"[bold red]Error loading model: {e}[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Model loaded successfully![/bold green]")
    console.print(f"[bold blue]Processing {image}...[/bold blue]")

    try:
        result = predictor.predict(str(image), return_confidence=show_confidence)
    except Exception as e:
        console.print(f"[bold red]Error during prediction: {e}[/bold red]")
        raise typer.Exit(1)

    # Display results
    console.print("\n[bold green]═══ Recognition Result ═══[/bold green]")
    console.print(f"\n{result['text']}\n", style="bold cyan", justify="right")

    if show_confidence:
        console.print(f"Confidence: {result.get('confidence', 0):.2%}", style="green")

    console.print(f"Processing time: {result['processing_time']:.3f}s", style="dim")

    # Save to file if requested
    if output:
        output.write_text(result["text"], encoding="utf-8")
        console.print(f"\n[green]Results saved to {output}[/green]")


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Directory containing images"),
    output: Path = typer.Option("results.json", help="Output JSON file"),
    model: Path = typer.Option("models/best_model.pth", help="Path to model checkpoint"),
    batch_size: int = typer.Option(8, help="Batch size"),
    pattern: str = typer.Option("*.jpg", help="File pattern to match"),
):
    """
    Process multiple images in a directory.
    """
    from src.inference import ArabicOCRPredictor

    # Find images
    image_files = list(input_dir.glob(pattern))

    if not image_files:
        console.print(f"[bold red]No images found matching {pattern} in {input_dir}[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold blue]Found {len(image_files)} images[/bold blue]")
    console.print(f"[bold blue]Loading model...[/bold blue]")

    predictor = ArabicOCRPredictor(model_path=str(model))

    console.print(f"[bold green]Model loaded![/bold green]")
    console.print(f"[bold blue]Processing images...[/bold blue]\n")

    # Process
    results = []
    for i in track(range(0, len(image_files), batch_size), description="Processing"):
        batch_files = image_files[i : i + batch_size]
        batch_results = predictor.predict_batch([str(f) for f in batch_files], batch_size)

        for file, result in zip(batch_files, batch_results):
            results.append({
                "file": str(file),
                "text": result["text"],
            })

    # Save results
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    console.print(f"\n[bold green]✓ Processed {len(results)} images[/bold green]")
    console.print(f"[green]Results saved to {output}[/green]")

    # Display sample
    console.print("\n[bold]Sample Results:[/bold]")
    table = Table()
    table.add_column("File", style="cyan")
    table.add_column("Text", style="green", justify="right")

    for result in results[:5]:
        table.add_row(
            Path(result["file"]).name,
            result["text"][:50] + "..." if len(result["text"]) > 50 else result["text"]
        )

    console.print(table)


@app.command()
def train(
    config: Path = typer.Option("configs/training_config.yaml", help="Training config"),
    model_config: Path = typer.Option("configs/model_config.yaml", help="Model config"),
    output_dir: Path = typer.Option("models/checkpoints", help="Output directory"),
):
    """
    Train the OCR model.
    """
    from src.training.train import train as train_model

    console.print("[bold blue]Starting training...[/bold blue]")

    try:
        train_model(
            model_config_path=str(model_config),
            train_config_path=str(config),
            output_dir=str(output_dir),
        )
    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def generate_data(
    num_samples: int = typer.Option(1000, help="Number of samples to generate"),
    output_dir: Path = typer.Option("data/augmented", help="Output directory"),
    split: str = typer.Option("train", help="Dataset split (train/val/test)"),
):
    """
    Generate synthetic training data.
    """
    from src.data.synthetic_generator import SyntheticArabicGenerator

    console.print(f"[bold blue]Generating {num_samples} samples...[/bold blue]")

    generator = SyntheticArabicGenerator()
    generator.generate_dataset(
        num_samples=num_samples,
        output_dir=str(output_dir),
        split=split,
    )

    console.print(f"[bold green]✓ Generated {num_samples} samples![/bold green]")
    console.print(f"[green]Saved to {output_dir}/{split}[/green]")


@app.command()
def server(
    host: str = typer.Option("0.0.0.0", help="Host address"),
    port: int = typer.Option(8000, help="Port number"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
):
    """
    Start the API server.
    """
    import uvicorn
    from src.api.app import app as fastapi_app

    console.print(f"[bold blue]Starting API server on {host}:{port}...[/bold blue]")
    console.print(f"[cyan]API docs: http://{host}:{port}/docs[/cyan]")

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def ui(
    port: int = typer.Option(8501, help="Port number"),
):
    """
    Start the Streamlit UI.
    """
    import subprocess

    console.print(f"[bold blue]Starting web UI on port {port}...[/bold blue]")
    console.print(f"[cyan]Opening http://localhost:{port}[/cyan]")

    subprocess.run([
        "streamlit",
        "run",
        "src/ui/streamlit_app.py",
        "--server.port",
        str(port),
    ])


if __name__ == "__main__":
    app()
