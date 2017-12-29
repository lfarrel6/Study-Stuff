This is an example which motivates the use of phantom types, by using them to prevent interchanging of Distance units.

Source: https://stackoverflow.com/questions/28247543/motivation-behind-phantom-types/28247968#28247968

There are three implementations in this file, so the pragmas are not all for the initial phantom type implementation

> {-# LANGUAGE GeneralizedNewtypeDeriving #-}
> {-# LANGUAGE KindSignatures             #-}
> {-# LANGUAGE DataKinds                  #-}
> {-# LANGUAGE GADTs                      #-}
> {-# LANGUAGE StandaloneDeriving         #-}

We create a newtype, Distance, which has a phantom type a. This makes the constructor for Distance have the type 'Double -> Distance a'. The type signature shows that the phantom type a is embedded into the Distance type. This has some useful features as we will see.


> newtype Distance a = Distance Double
>   deriving (Num, Show)


We declare two new data types, which will be used to give context to this phantom type example.


> data Kilometer
> data Mile


The type signature of the marathonDistance function shows that we will return something of type 'Distance Kilometer', and the body of the function simply provides a double to the Distance constructor.

This creates an instance of the Distance type, with the phantom type of Kilometer. This cannot interact with Miles, so our Distance language is quite type-safe


> marathonDistance :: Distance Kilometer
> marathonDistance = Distance 42.195


The type signature of this function restricts the parameter type and specifies the return type. If we were to call this with a value of type Distance Mile, it would be rejected at compile time.


> distanceKmToMiles :: Distance Kilometer -> Distance Mile
> distanceKmToMiles (Distance km) = Distance (0.621371 * km)

> marathonDistanceInMiles :: Distance Mile
> marathonDistanceInMiles = distanceKmToMiles marathonDistance


Further improvements on this system?

Using DataKinds....

We create a datatype which captures all valid units of measurement for the system.


> data LengthUnit = Kilometer1 | Mile1


And then restrict the phantom type a to be of type LengthUnit.


> newtype Distance1 (a :: LengthUnit) = Distance1 Double
>   deriving (Num, Show)


Another way to do this?

Using GADTs

> data Kilometer2
> data Mile2

> data Distance2 a where
>   KilometerDistance :: Double -> Distance2 Kilometer2
>   MileDistance :: Double -> Distance2 Mile2

> deriving instance Show (Distance2 a)

> marathonDistance2 :: Distance2 Kilometer2
> marathonDistance2 :: KilometerDistance 42.195