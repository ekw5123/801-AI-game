# proof of concept, no longer need this

import random
#look into pygame for gui
# board the user cannot see
board = [[0,0,0,0,0,0,0,0,0,0], #0 = no bomb, 
        [0,0,0,0,0,0,0,0,0,0],  #1 = bomb
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],]
# board user can see
boardDisplay=[[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1], #-1 is unknown
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        ]

TotalRows=len(boardDisplay)
TotalCols=len(boardDisplay[0])
numMines = 15

def checkMinesAround (row,col):
    totalAroundSpot=0
    r=row-1
    while r<= row+1:
        if r>=0 and r<TotalRows:
            c=col-1
            while c<=col+1:
                if c>=0 and c<TotalCols:
                    totalAroundSpot=totalAroundSpot+board[r][c]
                c=c+1
        r=r+1
    return totalAroundSpot

#add mines


def defineMines(startRow,startCol):
    num=0 #num active mines
    while num < numMines:
        row = random.randint(0,TotalRows-1)
        col = random.randint(0,TotalCols-1)

        if board[row][col] == 0:
            if (row is not startRow) and (col is not startCol):
                board[row][col]=1 #addmine
                num+=1


def displaySol():
    for row in range(0,TotalRows):
        for col in range(0,TotalCols):
            print(board[row][col],end=" ")
        print("")



def displayBoard():
    print("-"*50)
    for row in range(0,TotalRows):
        print(" ", end = " | ")
        for col in range(0,TotalCols):
            if boardDisplay[row][col]==-1:
                print(" ", end = " | ")
            else:
                print(boardDisplay[row][col],end=" | ")
        print("")
        print("-"*50)
# displaySol() #for debugging
displayBoard() #for debugging

def is_int(input_string):
    try:
        int(input_string)
        return True
    except ValueError:
        return False

def requestinput():
    ask1 = input("guess a row(1," +str(TotalRows)+"): ")
    while is_int(ask1) is not True:
        ask1 = input("that's not an integer, guess a row(1," +str(TotalRows)+"): ")
    while int(ask1)<0 or int(ask1)>TotalRows:
        ask1 = input("that's not on the grid, guess a row(1," +str(TotalRows)+"): ")
    row=int(ask1)-1
    ask1 = input("guess a col(1,"+str(TotalCols)+"): ")
    while is_int(ask1) is not True:
        ask1 = input("that's not an integer, guess a col(1," +str(TotalCols)+"): ")
    while int(ask1)<0 or int(ask1)>TotalCols:
        ask1 = input("that's not on the grid, guess a col(1," +str(TotalCols)+"): ")
    col=int(ask1)-1
    return row,col


guess = 0
row,col=requestinput()
defineMines(row,col)

if board[row][col]==1:
    print("boom you died")
    displaySol()
    guess=TotalRows*TotalCols
else:
    boardDisplay[row][col]=checkMinesAround(row,col)
    displayBoard()
    guess+=1
while guess < (TotalRows*TotalCols-numMines):
    # row=int(input("guess a row(1," +str(TotalRows)+"): "))-1
    # col=int(input("guess a col(1,"+str(TotalCols)+"): "))-1
    row,col=requestinput()
    if board[row][col]==1:
        print("boom you died")
        displaySol()
        break
    else:
        boardDisplay[row][col]=checkMinesAround(row,col)
        displayBoard()
        print("you win!")
    guess+=1
