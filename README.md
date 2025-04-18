# Blog Generation System

A powerful AI-driven blog generation system that creates SEO-optimized blog posts using multiple specialized agents.

## Features

- **AI-Powered Content Generation**: Uses advanced language models to generate high-quality blog content
- **SEO Optimization**: Automatically optimizes content for search engines
- **Multi-Agent Architecture**: Specialized agents for different aspects of blog creation
- **Command Line Interface**: Easy-to-use CLI for blog generation
- **Web Interface**: Streamlit-based web interface for interactive blog generation
- **Metadata Management**: Comprehensive SEO metadata generation and management
- **File Export**: Exports content in markdown format with associated metadata

## Architecture

The system uses a multi-agent architecture with the following components:

1. **BlogEngine**: Main orchestrator that coordinates the blog generation process
2. **WritingAgent**: Generates high-quality blog content
3. **SEOAgent**: Optimizes content for search engines
4. **ExecutionAgent**: Handles file export and metadata management
5. **AgentCoordinator**: Coordinates workflow between different agents

## Prerequisites

- Python 3.8+
- Required API keys:
  - `GROQ_API_KEY`
  - `NEWSDATA_API_KEY`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BlogAgent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```env
GROQ_API_KEY=your_groq_api_key
NEWSDATA_API_KEY=your_newsdata_api_key
```

## Usage

### Command Line Interface

The CLI provides a flexible way to generate blog posts with various options.

#### Basic Examples

```bash
# Generate a blog about Python programming
python cli.py "Python programming best practices"

# Generate a travel blog
python cli.py "Top 10 travel destinations in 2024"

# Generate a technical tutorial
python cli.py "Getting started with Docker containers"
```

#### Advanced Usage

```bash
# Generate a professional blog with custom output directory
python cli.py "Cloud computing trends" --tone professional --output-dir ./tech-blogs

# Generate a casual blog with verbose output
python cli.py "Healthy eating habits" --tone casual --verbose

# Generate an educational blog about history
python cli.py "Ancient civilizations" --tone educational --output-dir ./history-blogs

# Generate a technical blog with specific tone
python cli.py "Machine learning algorithms" --tone technical --verbose
```

#### Batch Processing

```bash
# Enter batch mode
python cli.py --batch

# Then enter topics one per line:
Python programming basics
Web development trends
Data science fundamentals
# Press Ctrl+D when done
```

#### Output Examples

The CLI will generate output like this:
```
Generating blog post for topic: Python programming best practices
Tone: professional
Output directory: ./output

=== Blog Generation Results ===
Title: 10 Essential Python Programming Best Practices for 2024
Reading Time: 8 minutes
Keywords: python, programming, best practices, code quality, development

Readability Scores:
Flesch Reading Ease: 65.2
Flesch-Kincaid Grade Level: 8.1

Generated Files:
- Markdown: ./output/python-programming-best-practices.md
- Metadata: ./output/python-programming-best-practices_meta.json
```

### BlogEngine Demo

To run the BlogEngine demo with a specific topic:
```bash
python BlogEngineDemo.py "your topic here"
```

Example with different tones:
```bash
# Professional tone
python BlogEngineDemo.py "Python programming best practices" --tone professional

# Casual tone
python BlogEngineDemo.py "Top 10 travel destinations" --tone casual

# Educational tone
python BlogEngineDemo.py "Machine learning basics" --tone educational
```

The demo will:
1. Generate a complete blog post
2. Show the generation process in real-time
3. Display the final output with SEO metadata
4. Save the generated content to the output directory

### Web Interface

Run the web interface:
```bash
streamlit run web_interface.py
```

## File Naming Convention

The system generates filenames based on SEO metadata following these rules:

1. Uses SEO-generated slug as the primary source
2. Falls back to title if no slug is available
3. Converts to lowercase and replaces spaces with hyphens
4. Removes special characters while preserving hyphens
5. Ensures reasonable filename length
6. Adds appropriate file extensions (.md for content, _meta.json for metadata)

Example filenames:
```
python-best-practices.md
python-best-practices_meta.json
```

## Output Structure

The system generates two files for each blog post:

1. **Markdown File** (`*.md`):
   - Contains the formatted blog content
   - Includes headers, sections, and formatting

2. **Metadata File** (`*_meta.json`):
   - Contains SEO metadata
   - Includes title, description, keywords
   - Stores reading time and other metrics

