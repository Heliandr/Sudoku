# Sudoku
Sudoku solver with OCR and GUI. 

This is a Python application that can recognize Sudoku puzzles from an image using OCR (Optical Character Recognition), and then solve them using a backtracking algorithm. It features a Tkinter-based GUI for easy interaction.


Features: 
Load a Sudoku image and extract digits using a CNN + MLP-based OCR system.
Manual editing of puzzle entries available.
Solve puzzles using a classic backtracking algorithm.


Requirements:
Python 3.8+
Packages: torch, numpy, opencv-puthon, tkinter

How it works: 
OCR:
A Convolutional Neural Network (CNN) and a Multi-Layer Perceptron (MLP) both try to recognize the digit in each cell.
If both models agree or the CNN is confident, the digit is accepted.

Solver:
Uses recursive backtracking to fill in the remaining cells while checking Sudoku rules.

GUI:
Built using tkinter, allowing users to interact easily with the puzzle, load images, and see results.


Here are some pictures illustrating the applications appearance and how it works 
as you can see there is a menu on the top left with the options load and detect, solve, clear and exit.
<img width="361" height="434" alt="image" src="https://github.com/user-attachments/assets/0e36c721-4c7e-49ad-8c59-7aecac4999a6" />



here is a sudoku puzzle selected as an example
<img width="849" height="1131" alt="image" src="https://github.com/user-attachments/assets/9174f3e0-17fa-4537-931b-142bf9cae33f" />




ypu can see the accuracy of detection. some numbers are wrong or not recognized, you can manually change them.
<img width="361" height="434" alt="image" src="https://github.com/user-attachments/assets/c50126b0-9d67-4835-affb-d1bef1ecc3ba" />




The result after solution: 
<img width="371" height="441" alt="image" src="https://github.com/user-attachments/assets/71ea2e23-38f2-405c-8350-529ccaad58fa" />


