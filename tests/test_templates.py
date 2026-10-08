from pathlib import Path

from neuromind.templates import HandlebarsTemplateEngine


def test_inline_handlebars_rendering():
    engine = HandlebarsTemplateEngine()
    rendered = engine.render(
        "Hello {{name}}!{{#if context}} Context: {{context}}{{/if}}",
        {"name": "NeuroMind", "context": "research"},
    )
    assert rendered == "Hello NeuroMind! Context: research"


def test_file_rendering():
    root = Path(__file__).parents[1] / "templates"
    engine = HandlebarsTemplateEngine(root)
    rendered = engine.render_file(
        "Neuromindai.handlebars",
        {
            "task": "Explain attention.",
            "instructions": "Be concise.",
            "response_format": "paragraph",
        },
    )
    assert "Explain attention." in rendered
    assert "Be concise." in rendered
    assert "Context:" not in rendered
