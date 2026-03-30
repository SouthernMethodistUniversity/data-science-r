##-----------------------##
## 3 Data Types - R Code ##
##-----------------------##

#### Identification ####

x <- 2 # reassign your variable if you need to
class(x) # call the class() function on x

#### Numbers ####

#### The numeric data type ####

class(2)
class(3.14)
class(-0.4621473)

round(3.56)
floor(3.56)

#### The integer data type ####

i <- as.integer(5)
i

class(i)
class(5)

#### The complex data type ####

c <- 3+2i
c
class(c)

complex(real = 3, imaginary = 2)
as.complex(3+2i)

#### Special Numbers and Functions ####

pi

exp(1) # e = e^1
exp(5) # e^5

log(2)
log(2, base = 10)
log(2) == log(2, base = exp(1))

#### The logical data type ####

x <- TRUE # explicitly
class(x)
x

y <- 1 == 1 # boolean comparisons
y

z <- "apples" == "oranges"
z

x + y + z # TRUE + TRUE + FALSE = 1 + 1 + 0

#### The character data type ####

x <- "apple"
class(x)
x

#### Coercion ####

x <- 2
class(x)
x

y <- as.character(x)
class(y)
y

as.integer()
as.numeric()
as.logical()

as.factor()
as.vector() 
as.data.frame()