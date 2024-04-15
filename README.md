# Tube2Slide

## Overview
Tube2Slide is a Python-based tool that summarizes YouTube videos. It is designed as a part of the coursework for Assignment 1 / Exercise 3. This tool utilizes text OCR (Optical Character Recognition) to extract subtitles from videos and generate concise summaries.

## Important Notes

### Compatibility and Setup

- **Anaconda on Windows:** If you encounter a multiprocessing (MP) error, execute the following command in the Anaconda terminal (PowerShell) to resolve it:
  ```
  $env:KMP_DUPLICATE_LIB_OK="TRUE"
  ```

### Performance and Customization

- **Text OCR Component:** The OCR component is resource-intensive and may increase computation time significantly. If necessary, you can disable this feature by commenting out the relevant function in the `main` script.

### Requirements

- **Virtual Environment:** A virtual environment is recommended for running Tube2Slide to manage dependencies effectively. All required packages are listed in the `requirements.txt` file. Please set up the virtual environment and install the dependencies.
