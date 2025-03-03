import tkinter as tk
from functools import partial

# Unicode dictionary for chess pieces
uniDict = {
    'WHITE': {'Pawn': "♙", 'Rook': "♖", 'Knight': "♘", 'Bishop': "♗", 'King': "♔", 'Queen': "♕"},
    'BLACK': {'Pawn': "♟", 'Rook': "♜", 'Knight': "♞", 'Bishop': "♝", 'King': "♚", 'Queen': "♛"}
}

# Initialize the chess board with Unicode symbols
board = [
    [uniDict['BLACK']['Rook'], uniDict['BLACK']['Knight'], uniDict['BLACK']['Bishop'], uniDict['BLACK']['Queen'], uniDict['BLACK']['King'], uniDict['BLACK']['Bishop'], uniDict['BLACK']['Knight'], uniDict['BLACK']['Rook']],
    [uniDict['BLACK']['Pawn']] * 8,
    [' '] * 8,
    [' '] * 8,
    [' '] * 8,
    [' '] * 8,
    [uniDict['WHITE']['Pawn']] * 8,
    [uniDict['WHITE']['Rook'], uniDict['WHITE']['Knight'], uniDict['WHITE']['Bishop'], uniDict['WHITE']['Queen'], uniDict['WHITE']['King'], uniDict['WHITE']['Bishop'], uniDict['WHITE']['Knight'], uniDict['WHITE']['Rook']]
]

turn = 'W'  # 'W' for White, 'B' for Black
selected = None
color1, color2 = 'white', 'gray'  # Board colors
game_over = False

# Helper to identify piece type from Unicode symbol
def get_piece_type(piece):
    if piece == uniDict['WHITE']['Pawn'] or piece == uniDict['BLACK']['Pawn']:
        return 'p'
    elif piece == uniDict['WHITE']['Rook'] or piece == uniDict['BLACK']['Rook']:
        return 'r'
    elif piece == uniDict['WHITE']['Knight'] or piece == uniDict['BLACK']['Knight']:
        return 'n'
    elif piece == uniDict['WHITE']['Bishop'] or piece == uniDict['BLACK']['Bishop']:
        return 'b'
    elif piece == uniDict['WHITE']['Queen'] or piece == uniDict['BLACK']['Queen']:
        return 'q'
    elif piece == uniDict['WHITE']['King'] or piece == uniDict['BLACK']['King']:
        return 'k'
    return None

# Check if piece is White
def is_white_piece(piece):
    return piece in uniDict['WHITE'].values()

# Find the king's position
def find_king(board, player):
    king = uniDict['WHITE']['King'] if player == 'W' else uniDict['BLACK']['King']
    for i in range(8):
        for j in range(8):
            if board[i][j] == king:
                return i, j
    return None

# Check if a square is attacked by the opponent
def is_square_attacked(i, j, attacker, board):
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece != ' ' and ((attacker == 'B' and not is_white_piece(piece)) or (attacker == 'W' and is_white_piece(piece))):
                if is_valid_move(r, c, i, j, board, attacker, ignore_king=True):
                    return True
    return False

# Check if the player is in check
def is_in_check(player, board):
    king_pos = find_king(board, player)
    if king_pos is None:
        return False
    attacker = 'B' if player == 'W' else 'W'
    return is_square_attacked(king_pos[0], king_pos[1], attacker, board)

# Check if the player has any legal moves
def has_legal_moves(player, board):
    for start_i in range(8):
        for start_j in range(8):
            piece = board[start_i][start_j]
            if piece != ' ' and ((player == 'W' and is_white_piece(piece)) or (player == 'B' and not is_white_piece(piece))):
                for end_i in range(8):
                    for end_j in range(8):
                        if is_valid_move(start_i, start_j, end_i, end_j, board, player) and not leaves_king_in_check(start_i, start_j, end_i, end_j, board, player):
                            return True
    return False

# Check for game-over conditions
def check_game_over():
    global game_over
    opponent = 'B' if turn == 'W' else 'W'
    if is_in_check(opponent, board):
        if not has_legal_moves(opponent, board):
            game_over = True
            winner = 'White' if turn == 'W' else 'Black'
            turn_label.config(text=f"Checkmate! {winner} wins!")
            return
    elif not has_legal_moves(opponent, board):
        game_over = True
        turn_label.config(text="Stalemate! Draw.")
        return
    # Update turn label if no game-over condition
    turn_label.config(text=f"{'Black' if turn == 'B' else 'White'}'s turn" + (" - Check!" if is_in_check(opponent, board) else ""))

# Move validation
def is_valid_move(start_i, start_j, end_i, end_j, board, turn, ignore_king=False):
    piece = board[start_i][start_j]
    if piece == ' ' or (turn == 'W' and not is_white_piece(piece)) or (turn == 'B' and is_white_piece(piece)):
        return False
    dest_piece = board[end_i][end_j]
    if dest_piece != ' ' and ((is_white_piece(piece) and is_white_piece(dest_piece)) or (not is_white_piece(piece) and not is_white_piece(dest_piece))):
        return False
    piece_type = get_piece_type(piece)
    if piece_type == 'p':
        return is_valid_pawn_move(start_i, start_j, end_i, end_j, board, is_white_piece(piece))
    elif piece_type == 'r':
        return is_valid_rook_move(start_i, start_j, end_i, end_j, board)
    elif piece_type == 'n':
        return is_valid_knight_move(start_i, start_j, end_i, end_j, board)
    elif piece_type == 'b':
        return is_valid_bishop_move(start_i, start_j, end_i, end_j, board)
    elif piece_type == 'q':
        return is_valid_queen_move(start_i, start_j, end_i, end_j, board)
    elif piece_type == 'k' and not ignore_king:
        return is_valid_king_move(start_i, start_j, end_i, end_j, board)
    return False

def is_valid_pawn_move(start_i, start_j, end_i, end_j, board, is_white):
    direction = -1 if is_white else 1
    start_row = 6 if is_white else 1
    if start_j == end_j:  # Forward move
        if end_i == start_i + direction and board[end_i][end_j] == ' ':
            return True
        if start_i == start_row and end_i == start_i + 2 * direction and board[start_i + direction][start_j] == ' ' and board[end_i][end_j] == ' ':
            return True
    elif abs(end_j - start_j) == 1 and end_i == start_i + direction:  # Diagonal capture
        if board[end_i][end_j] != ' ':
            return True
    return False

def is_valid_rook_move(start_i, start_j, end_i, end_j, board):
    if start_i == end_i:  # Horizontal
        step = 1 if end_j > start_j else -1
        for j in range(start_j + step, end_j, step):
            if board[start_i][j] != ' ':
                return False
        return True
    elif start_j == end_j:  # Vertical
        step = 1 if end_i > start_i else -1
        for i in range(start_i + step, end_i, step):
            if board[i][start_j] != ' ':
                return False
        return True
    return False

def is_valid_knight_move(start_i, start_j, end_i, end_j, board):
    di, dj = abs(end_i - start_i), abs(end_j - start_j)
    return (di == 2 and dj == 1) or (di == 1 and dj == 2)

def is_valid_bishop_move(start_i, start_j, end_i, end_j, board):
    if abs(end_i - start_i) == abs(end_j - start_j):
        steps = abs(end_i - start_i)
        di = 1 if end_i > start_i else -1
        dj = 1 if end_j > start_j else -1
        for k in range(1, steps):
            if board[start_i + k * di][start_j + k * dj] != ' ':
                return False
        return True
    return False

def is_valid_queen_move(start_i, start_j, end_i, end_j, board):
    return is_valid_rook_move(start_i, start_j, end_i, end_j, board) or is_valid_bishop_move(start_i, start_j, end_i, end_j, board)

def is_valid_king_move(start_i, start_j, end_i, end_j, board):
    di, dj = abs(end_i - start_i), abs(end_j - start_j)
    return di <= 1 and dj <= 1 and (di + dj > 0)

# Check if a move leaves the king in check
def leaves_king_in_check(start_i, start_j, end_i, end_j, board, turn):
    temp_board = [row[:] for row in board]
    temp_board[end_i][end_j] = temp_board[start_i][start_j]
    temp_board[start_i][start_j] = ' '
    return is_in_check(turn, temp_board)

# Handle square clicks
def on_click(i, j):
    global selected, turn, game_over
    if game_over:
        return
    piece = board[i][j]
    if selected is None:
        if piece != ' ' and ((turn == 'W' and is_white_piece(piece)) or (turn == 'B' and not is_white_piece(piece))):
            selected = (i, j)
            buttons[i][j].config(bg='yellow')
    else:
        sel_i, sel_j = selected
        sel_piece = board[sel_i][sel_j]
        if is_valid_move(sel_i, sel_j, i, j, board, turn) and not leaves_king_in_check(sel_i, sel_j, i, j, board, turn):
            # Move the piece
            board[i][j] = sel_piece
            board[sel_i][sel_j] = ' '
            buttons[i][j].config(text=sel_piece)
            buttons[sel_i][sel_j].config(text=' ')
            # Reset colors
            buttons[sel_i][sel_j].config(bg=color1 if (sel_i + sel_j) % 2 == 0 else color2)
            buttons[i][j].config(bg=color1 if (i + j) % 2 == 0 else color2)
            selected = None
            # Switch turn
            turn = 'B' if turn == 'W' else 'W'
            check_game_over()
        else:
            buttons[sel_i][sel_j].config(bg=color1 if (sel_i + sel_j) % 2 == 0 else color2)
            if piece != ' ' and ((turn == 'W' and is_white_piece(piece)) or (turn == 'B' and not is_white_piece(piece))):
                selected = (i, j)
                buttons[i][j].config(bg='yellow')
            else:
                selected = None

# Set up the GUI
root = tk.Tk()
root.title("Chess with Checkmate Detection")
frame = tk.Frame(root)
frame.grid()
buttons = [[None for _ in range(8)] for _ in range(8)]
for i in range(8):
    for j in range(8):
        bg_color = color1 if (i + j) % 2 == 0 else color2
        btn = tk.Button(frame, text=board[i][j], width=2, height=1, bg=bg_color, font=("Arial", 20))
        btn.grid(row=i, column=j)
        buttons[i][j] = btn
        btn.config(command=partial(on_click, i, j))
turn_label = tk.Label(root, text="White's turn")
turn_label.grid(row=8, column=0, columnspan=8)
root.mainloop()