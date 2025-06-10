This repository contains code for several fast Operations Research-related Gymnax environments.

RidesharePoolDispatch makes one key approximation for speed: we assume that cars, once en route to a waypoint, cannot be diverted -- new waypoints assigned must occur after current one. This enables us to reduce a car's state representation to just its current waypoint plan, as opposed to having to interpolate its location on every dispatch.

Currently there are only two waypoints per trip, making this quite a rough approximation. The approximation could be improved by specifying multiple waypoints per trip.




