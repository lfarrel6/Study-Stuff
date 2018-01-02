In this question you are to write a Haskell Domain Specific Language which implements "turtle graphics". In this system we imagine that there is a small robot (called the turtle) which holds a pen. As the turtle moves about on a sheet of paper the pen leaves a trail, and this is used to draw pictures.

The tutle understands only four primitive operations:
	- step  (which moves the turtle forward a single step)
	- right (which instructs the turtle to rotate 90 degrees right)
	- up    (which lifts teh pen from the paper - subsequent movement will not leave a trail)
	- down  (which lowers the pen to the paper - subsequent movement will leave a trail)

For this question you must implement an embedded DSL in Haskell which simulates the turtle graphics. The result of running a program in this DSL should be a list of line segments to draw (represented as pairs of points).
The library should allow for programs like this:

	`drawing = do
	   down; forward; right ; up; forward ; down ; right ; forward`
	  to produce a list of line segments:
	    [ ( (0,0), (0,1) ), ( (1,1), (1,0) ) ]

-----------------------------------------------------------------------------------------------

a) Give a definition for a Haskell data type Program which will form the basis for the turtle graphics monad, and which will allow Haskell programs such as the example above (i.e. make Program a monad).

-----------------------------------------------------------------------------------------------

create alias Point to refer to tuples of Ints for readability

> type Point = (Int,Int)

create Turtle data type to track the state of the pen, the current location and the direction

> data Turtle = Turtle { drawing         :: Bool
>                      , currentLocation :: Point
>                      , facing          :: Double
>                      }

State monad implementation of Program

> type Program = State (Turtle,[(Point,Point)]) ()

-----------------------------------------------------------------------------------------------

b) Provide implementations of the four primitive operations required of the turtle graphics language.

----------------------------------------------------------------------------------------------

------------------------
Implementation for right
------------------------

> right :: Program
> right = modify (\(turtle,lines) -> (changeDirection RotationAngle turtle, lines))
> right2 = enactChange changeDir

---------------------
Implementation for up
---------------------

> up :: Program
> up = modify (\(turtle,lines) -> (changeDrawing False turtle, lines))
> up2 = enactChange raise

-----------------------
Implementation for down
-----------------------

> down :: Program
> down = modify (\(turtle,lines) -> (changeDrawing False turtle, lines))
> down2 = enactChange lower

-----------------------
Implementation for forward
-----------------------

Actions required: 
	- take turtle from program
	- get currentLocation value and current facing (direction)
	- calculate new location value
	- if pen is down, append to list of points

---------------------------------------------------------------

> forward :: Program
> forward = do
>  (turtle,lines) <- get
>  let moved  = step turtle
>      oldLoc = (currentLocation turtle)
>      drawB  = (drawing turtle)
>      lines' = if drawB then (oldLoc,moved) : lines else lines
>  put (Turtle drawB moved (facing turtle), lines')

---------------------------
Auxiliary Functions & Types
---------------------------

Higher Order Function to perform changes on the Turtle

> enactChange :: ( Turtle -> Turtle ) -> Program
> enactChange f = modify (\(turtle,lines) -> (f turtle, lines))

Function to set state of the pen

> changeDrawing :: Bool -> Turtle -> Turtle
> changeDrawing b t@Turtle{..} = Turtle b currentLocation facing

H.O.F Friendly drawing modifiers

> raise :: Turtle -> Turtle
> raise t@Turtle{..} = Turtle False currentLocation facing

> lower :: Turtle -> Turtle
> lower t@Turtle{..} = Turtle True currentLocation facing

Function to change the direction the turtle is facing

> changeDirection :: Double -> Turtle -> Turtle
> changeDirection a t@Turtle{..} = Turtle drawing currentLocation (facing + a) 

H.O.F Friendly direction modifier

> changeDir :: Turtle -> Turtle
> changeDir t@Turtle{..} = Turtle drawing currentLocation (facing + RotationAngle)

Function to move the turtle forward one step

> step :: Turtle -> Point
> step t@Turtle{..} = (oldX+dX, oldY+dY)
>   where
>    (oldX,oldY) = currentLocation
>    reducedAngle = facing / RotationAngle
>    (dX,dY) = case ( interpret $ floor $ reducedAngle ) of
>                   North -> (0,1)
>                   East  -> (1,0)
>                   South -> (0,-1)
>                   West  -> (-1,0)
>                   _     -> (0,0)
>   interpret a = a `mod` 4

Standard Unit to change direction

> type RotationAngle = 90

Type aliases for Compass points (readability)

> type North = 0
> type East  = 1
> type South = 2
> type West  = 3

----------------------------------------------------------------------------------------------

c) Make Program an instance of Monad.

----------------------------------------------------------------------------------------------

The provided implementation utilises the State monad

----------------------------------------------------------------------------------------------

d) Provice a `draw` function with the following type (assume the turtle begins at the origin facing north).
	`draw :: Program -> [Line]`

----------------------------------------------------------------------------------------------

Need to runState and extract list, and reverse (as we pushed values in reverse)

> draw :: Program -> [(Point,Point)]
> draw p = reverse $ snd . snd $ runState p (Turtle False (0,0) 0, [])

----------------------------------------------------------------------------------------------

e) Give a definition function repeat which will run a turtle graphics program repeatedly so that a square could be drawn via the expression `repeat 4 (do forward ; right)`.

----------------------------------------------------------------------------------------------

repeat :: Int -> Program -> Program
repeat n p = foldl1 (>>) $ replicate n p


----------------------------------------------------------------------------------------------

f) We could extend the set of primitives to allow for more than four possible facings; if we change the definition of right so that it rotates the turtle only 45 degrees right then there will be eight possible facings.
Taking your existing solution indicate systematically where it would need to change in order to support this change. You do not need to provide the full implementation, but explain any assumptions or restrictions that are relevant to your answer.

----------------------------------------------------------------------------------------------

The value aliased as RotationAngle would obviously need to be changed from 90 to 45. The arithmetic used in step would also need to change. The revised version would look something like this (assuming RotationAngle has been changed to 45):

> step :: Turtle -> Point
> step t@Turtle{..} = (oldX+dX, oldY+dY)
>   where
>    (oldX,oldY) = currentLocation
>    reducedAngle = facing / RotationAngle
>    (dX,dY) = case ( interpret $ floor $ reducedAngle ) of
>                   N  -> (0,1)
>                   NE -> (1,1)
>                   E  -> (1,0)
>                   SE -> (1,-1)
>                   S  -> (0,-1)
>                   SW -> (-1,-1)
>                   W  -> (-1,0)
>                   NW -> (-1,1)
>                   _  -> (0,0)
>   interpret a = a `mod` 8
>
> type N  = 0
> type NE = 1
> type E  = 2
> type SE = 3
> type S  = 4
> type SW = 5
> type W  = 6
> type NW = 7

Note that the additional aliases for the compass points are only for readability.