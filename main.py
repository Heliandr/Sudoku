import os
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tkinter import *
from tkinter import filedialog, messagebox






PROB_THRESHOLD = 0.75
INCLUDE_ZERO_CLASS = False
SHOW_DEBUG_PRINTS = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CNN_WEIGHTS = "weights_pytorch.pth"
DEFAULT_MLP_WEIGHTS = "weights.npz"


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x



def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def order_points(pts):
    pts = sorted(pts, key=lambda p: p[1])  # sort by y
    top = pts[:2]
    bottom = pts[2:]
    top_left, top_right = sorted(top, key=lambda p: p[0])
    bottom_left, bottom_right = sorted(bottom, key=lambda p: p[0])
    return [top_left, top_right, bottom_right, bottom_left]

def warp_sudoku(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(gray, 150, 200)

    contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        raise ValueError("No contours found.")

    biggest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, epsilon, True)

    if len(approx) != 4:
        raise ValueError("Sudoku outline not a quadrilateral (found %d points)." % len(approx))

    corners = np.array(order_points(approx.reshape(-1, 2)), dtype="float32")
    max_width = int(max(distance(corners[0], corners[1]), distance(corners[2], corners[3])))
    max_height = int(max(distance(corners[1], corners[2]), distance(corners[0], corners[3])))

    dst = np.array([[0, 0],
                    [max_width - 1, 0],
                    [max_width - 1, max_height - 1],
                    [0, max_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(gray, M, (max_width, max_height), flags=cv2.INTER_LANCZOS4)
    warped = cv2.resize(warped, (252, 252), interpolation=cv2.INTER_AREA)
    return warped.astype("float32") / 255.0



def mlp_forward(inp_vec, params):

    z1 = inp_vec @ params['w1'] + params['b1']
    a1 = np.maximum(0, z1)
    z2 = a1 @ params['w2'] + params['b2']
    a2 = np.maximum(0, z2)
    z3 = a2 @ params['w3'] + params['b3']
    a3 = np.maximum(0, z3)
    return a3


class DigitRecognizer:
    def __init__(self,
                 cnn_weights=DEFAULT_CNN_WEIGHTS,
                 mlp_weights=DEFAULT_MLP_WEIGHTS,
                 prob_threshold=PROB_THRESHOLD):
        self.prob_threshold = prob_threshold
        self.cnn = CNN()
        if not os.path.isfile(cnn_weights):
            raise FileNotFoundError(f"CNN weights file not found: {cnn_weights}")
        self.cnn.load_state_dict(torch.load(cnn_weights, map_location='cpu'))
        self.cnn.eval()

        if not os.path.isfile(mlp_weights):
            raise FileNotFoundError(f"MLP weights file not found: {mlp_weights}")
        mlp_params = np.load(mlp_weights)
        needed = {'w1','w2','w3','b1','b2','b3'}
        if not needed.issubset(mlp_params.files):
            raise ValueError("MLP weights file missing required keys.")
        self.mlp_params = {k: mlp_params[k] for k in needed}

    def preprocess_cells(self, warped):
        cells = []
        for r in range(9):
            for c in range(9):
                cell = warped[28*r:28*(r+1), 28*c:28*(c+1)]  # 28x28
                # Crop 3:25 region then pad with 1's (white)
                inner = cell[3:25, 3:25]
                padded = np.pad(inner, ((3,3),(3,3)), mode='constant', constant_values=1)
                # Invert style as your code did
                img = np.zeros((28,28), dtype=np.float32)
                # Vectorized equivalent of nested loop:
                img[padded <= 0.5] = 1
                # padded > 0.5 remains 0
                cells.append(img)
        return np.array(cells)  # (81,28,28)

    def recognize(self, image_path):
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError("cv2.imread failed (None object).")

        warped = warp_sudoku(original)
        processed = self.preprocess_cells(warped)
        board = np.zeros((9,9), dtype=int)
        confidences = np.zeros((9,9), dtype=float)

        for idx in range(81):
            r, c = divmod(idx, 9)
            img_tensor = torch.tensor(processed[idx]).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
            with torch.no_grad():
                logits = self.cnn(img_tensor.to(device))
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_cnn = int(np.argmax(probs))
            conf_cnn = float(np.max(probs))

            # Classical MLP
            classical_out = mlp_forward(processed[idx].reshape(1,784), self.mlp_params)
            pred_mlp = int(np.argmax(classical_out))

            # Decision rule
            if (pred_cnn == pred_mlp) or (conf_cnn >= self.prob_threshold):
                digit = pred_cnn
                confidence = conf_cnn
            else:
                digit = 0
                confidence = 0.0

            # Optionally treat '0' as blank always
            if not INCLUDE_ZERO_CLASS and digit == 0:
                board[r, c] = 0
            else:
                board[r, c] = digit
            confidences[r, c] = confidence

            if SHOW_DEBUG_PRINTS:
                print(f"Cell ({r},{c}) CNN:{pred_cnn} conf:{conf_cnn:.2f} MLP:{pred_mlp} -> {board[r,c]}")

        return board, confidences


def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j
    return None

def is_valid(board, row, col, val):
    if any(board[row][k] == val for k in range(9)):
        return False
    if any(board[k][col] == val for k in range(9)):
        return False
    br, bc = 3 * (row//3), 3 * (col//3)
    for r in range(br, br+3):
        for c in range(bc, bc+3):
            if board[r][c] == val:
                return False
    return True

def solve_board(board):
    pos = find_empty(board)
    if not pos:
        return True
    r, c = pos
    for num in range(1, 10):
        if is_valid(board, r, c, num):
            board[r][c] = num
            if solve_board(board):
                return True
            board[r][c] = 0
    return False


class SudokuApp:
    def __init__(self, master):
        self.master = master
        master.title("Sudoku Solver")
        self.recognizer = None  # lazy init on first use
        self.cells = [[StringVar(master) for _ in range(9)] for _ in range(9)]
        self.entries = [[None for _ in range(9)] for _ in range(9)]
        self.confidences = np.zeros((9,9), dtype=float)
        self.create_grid()
        self.create_menu()


    def create_grid(self):
        font = ('Arial', 18)
        for r in range(9):
            for c in range(9):
                color = 'gray' if (r//3 + c//3) % 2 == 0 else 'white'
                e = Entry(self.master,
                          width=2,
                          font=font,
                          bg=color,
                          justify='center',
                          borderwidth=0,
                          highlightcolor='yellow',
                          highlightthickness=1,
                          highlightbackground='black',
                          textvariable=self.cells[r][c])
                e.grid(row=r, column=c, padx=1, pady=1)
                e.bind('<KeyRelease>', self._filter_input)
                self.entries[r][c] = e

    def create_menu(self):
        menubar = Menu(self.master)
        self.master.config(menu=menubar)
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load & Detect", command=self.action_load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Solve", command=self.action_solve)
        file_menu.add_command(label="Clear", command=self.action_clear)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.quit)


    def _filter_input(self, event):
        w = event.widget
        txt = w.get().strip()
        if txt == '':
            return

        if len(txt) > 1 or txt not in '123456789':
            w.delete(0, END)


    def get_board(self):
        board = []
        for r in range(9):
            row = []
            for c in range(9):
                v = self.cells[r][c].get().strip()
                if v.isdigit() and 1 <= int(v) <= 9:
                    row.append(int(v))
                else:
                    row.append(0)
            board.append(row)
        return board

    def set_board(self, board):
        for r in range(9):
            for c in range(9):
                val = board[r][c]
                self.cells[r][c].set(str(val) if val != 0 else '')

    def clear_confidence_highlight(self):
        for r in range(9):
            for c in range(9):
                base_color = 'gray' if (r//3 + c//3) % 2 == 0 else 'white'
                self.entries[r][c].config(fg='black', bg=base_color)




    def action_clear(self):
        self.clear_confidence_highlight()
        for r in range(9):
            for c in range(9):
                self.cells[r][c].set('')
        self.confidences[:] = 0.0

    def action_solve(self):
        board = self.get_board()
        if SHOW_DEBUG_PRINTS:
            print("Board sent to solver:")
            for row in board:
                print(row)

        working = [row[:] for row in board]
        if not self._board_is_consistent(working):
            messagebox.showerror("Invalid Puzzle")
            return
        if solve_board(working):
            self.set_board(working)
        else:
            messagebox.showerror("No solution found. check for mistake in entries.")

    def action_load_image(self):
        path = filedialog.askopenfilename(
            title="Select Sudoku Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All Files","*.*")]
        )
        if path:
            self._recognize_from_path(path, auto_solve=False)


    def action_recognize_and_solve(self):
        path = filedialog.askopenfilename(
            title="Select Sudoku Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All Files","*.*")]
        )
        if path:
            self._recognize_from_path(path, auto_solve=True)


    def _recognize_from_path(self, path, auto_solve=False):
        try:
            if self.recognizer is None:
                self.recognizer = DigitRecognizer()
            board, conf = self.recognizer.recognize(path)
            self.set_board(board)
            self.confidences = conf
        except Exception as e:
            messagebox.showerror("Recognition Error", str(e))
            return

        if auto_solve:
            board_for_solver = self.get_board()
            working = [row[:] for row in board_for_solver]
            if not self._board_is_consistent(working):
                messagebox.showerror("Invalid entry please correct")
                return
            if solve_board(working):
                self.set_board(working)
            else:
                messagebox.showwarning("No solution found, check for mistakes")


    def _board_is_consistent(self, board):
        # Rows
        for r in range(9):
            vals = [v for v in board[r] if v != 0]
            if len(vals) != len(set(vals)):
                return False
        # Columns
        for c in range(9):
            col_vals = [board[r][c] for r in range(9) if board[r][c] != 0]
            if len(col_vals) != len(set(col_vals)):
                return False
        # Subgrids
        for br in range(0,9,3):
            for bc in range(0,9,3):
                block = []
                for r in range(br, br+3):
                    for c in range(bc, bc+3):
                        v = board[r][c]
                        if v != 0:
                            block.append(v)
                if len(block) != len(set(block)):
                    return False
        return True



def main():
    root = Tk()
    root.resizable(False, False)
    app = SudokuApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()