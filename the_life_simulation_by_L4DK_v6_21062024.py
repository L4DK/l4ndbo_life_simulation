# File: the_life_simulation_by_L4DK_v6_21062024.py
# Author: Michael Landbo aka L4DK or L4ndbo or Landbo
# Date: 2024-06-21
# Version: 6_21062024
# ----------------------------------------------------------------------
# Description:
# This simulation models the evolution of life through the interactions
# of autonomous, AI-driven agents (atoms).
#
# The simulation includes various features such as:
#    - Realistic physics: gravity and Coulomb forces
#    - Collision detection and response
#    - Atom behavior: customizable properties, reproduction, mutation
#    - Visualization: energy levels, motion trails, color gradients
#    - User interaction: zoom, pan, time control, save/load states
#    - Evolutionary processes: life simulation, complex structures
#    - And MUCH more
#
# Relevant Links:
#   - Net Force and Gravity: https://en.wikipedia.org/wiki/Net_force
#   - Coulomb's Law: https://en.wikipedia.org/wiki/Coulomb%27s_law
#   - Evolution: https://en.wikipedia.org/wiki/Evolution
#   - Genetic Algorithm: https://en.wikipedia.org/wiki/Genetic_algorithm
#   - Collision Detection: https://en.wikipedia.org/wiki/Collision_detection
#   - Pygame Documentation: https://www.pygame.org/docs/
#   - Kinetic Energy: https://en.wikipedia.org/wiki/Kinetic_energy
#   - Simulation: https://en.wikipedia.org/wiki/Simulation
#   - Artificial Intelligence: https://en.wikipedia.org/wiki/Artificial_intelligence
# ----------------------------------------------------------------------

# Import libraries
import pygame  # for graphics
import random  # for random number generation
import math  # for math functions
import pickle  # for saving and loading
import time  # for timing

# Sets the MOST IMPORTANT rule in this simulation, which must be set to 42.
# The Answer to the Ultimate Question of Life, The Universe, and Everything
random.seed(42)

# Initialize Pygame
# Url: https://www.pygame.org/docs/index.html
pygame.init()  # initialize pygame

# Constants for the colors
BLACK = (0, 0, 0)  # Black
WHITE = (255, 255, 255)  # White
GREEN = (0, 255, 0)  # Green
RED = (255, 0, 0)  # Red
BLUE = (0, 0, 255)  # Blue
YELLOW = (255, 255, 0)  # Yellow
PURPLE = (255, 0, 255)  # Purple

ATOM_COLORS = [RED, YELLOW, BLUE, GREEN, WHITE, BLACK, PURPLE]
FOOD_COLORS = [GREEN, WHITE, BLACK, PURPLE]
# Constants for the colors
FOOD_COLOR = GREEN  # Color of food

# Constants for the simulation
WINDOW_SIZE = 1200  # Size of the window
NUM_ATOMS = 100  # Number of atoms
ATOM_SIZE = 5.0  # Size of the atoms
SPEED = 2.0  # Adjust initial speed if necessary
TRAIL_LENGTH = 10.0  # Length of motion trails
MAX_VELOCITY = 5.0  # Maximum velocity for atoms
DAMPING_FACTOR = 0.995  # Damping factor to reduce energy over time
COLLISION_DAMPING = 0.98  # Damping applied during collisions
FORCE_SCALE = 1.0  # Increased scale factor for visualizing forces
GRAVITY = 0.01  # Reduced gravity
COULOMB = 0.01  # Reduced Coulomb force

ZOOM_LEVEL = 1.0  # Initial zoom level
PAN_X = 0  # Initial pan position
PAN_Y = 0  # Initial pan position
FORCE_THRESHOLD = 100.0  # Distance threshold for force calculations
MATING_SEASON = True  # Enable mating season

# Constants for the time control
TIME_STEP = 0.1  # Time step for the simulation
SIMULATION_SPEED = 1.0  # Speed multiplier for the simulation

# Constants for the evolution
EVOLUTION_RATE = 0.01  # Rate of evolution

# Constants for the food
FOOD_RATE = 0.01  # Food generation rate
FOOD_SPAWN_PROBABILITY = 0.01  # Probability of food spawning

# Constants for the species
ATOM_SPECIES = "atom"
FOOD_SPECIES = "food"

# Constants for the battle
BATTLE_RADIUS = 10.0  # Radius of battle
BATTLE_FORCE_SCALE = 0.1  # Force scale for battle

# Constants for the reproduction
REPRODUCTION_CHANCE = 0.01  # Chance of reproduction

# Constants for the reproduction cooldown
REPRODUCTION_COOLDOWN = 60.0

# Constants for food distribution
NUM_FOOD_ATOMS = 50  # Number of food atoms
FOOD_SIZE = 3.0  # Size of food atoms


# Define the Atom class
class Atom:
    # Define the Atom constructor
    def __init__(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        mass: float,
        charge: float,
        size: float,
        color: tuple[int, int, int],
        species: str,
    ) -> None:
        """
        Initializes a new instance of the Atom class.

        Args:
            x (float): The x-coordinate of the atom.
            y (float): The y-coordinate of the atom.
            vx (float): The x-component of the atom's velocity.
            vy (float): The y-component of the atom's velocity.
            mass (float): The mass of the atom.
            charge (float): The charge of the atom.
            size (float): The size of the atom.
            color (tuple[int, int, int]): The color of the atom as a tuple of RGB values.
            species (str): The species of the atom.

        Returns:
            None

        Initializes the following instance variables:
            - x (float): The x-coordinate of the atom.
            - y (float): The y-coordinate of the atom.
            - vx (float): The x-component of the atom's velocity.
            - vy (float): The y-component of the atom's velocity.
            - mass (float): The mass of the atom.
            - charge (float): The charge of the atom.
            - size (float): The size of the atom.
            - color (tuple[int, int, int]): The color of the atom as a tuple of RGB values.
            - species (str): The species of the atom.
            - energy (int): The energy of the atom.
            - hunger (int): The hunger level of the atom.
            - health (int): The health level of the atom.
            - age (int): The age of the atom.
            - trail (list[tuple[float, float]]): The trail of the atom as a list of (x, y) coordinates.
            - fx (float): The x-component of the net force acting on the atom.
            - fy (float): The y-component of the net force acting on the atom.
            - in_battle (bool): Indicates whether the atom is in a battle.
            - reproduction_cooldown (float): The cooldown period for reproduction.
        """

        # Initialize the instance variables
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
        self.charge = charge
        self.size = size
        self.color = color
        self.species = species
        self.energy = 100
        self.hunger = 100
        self.health = 100
        self.age = 0
        self.trail: list[tuple[float, float]] = []
        self.fx = 0.0
        self.fy = 0.0
        self.in_battle: bool = False
        self.reproduction_cooldown: float = 0.0

    # REPRODUCTION
    # Reproduce the atom
    def reproduce(self, atoms):
        """
        Reproduce the atom by finding a mate and mating with it.

        Args:
            atoms (list): A list of Atom objects representing the population.

        Returns:
            None

        This function checks if the reproduction cooldown is zero.
        If it is, it finds a mate by calling the find_mate method.
        If a mate is found, it calls the mate_with method to mate with the mate
        and updates the reproduction cooldowns of both the atom and the mate.

        Note:
            - The find_mate method should be defined in the Atom class and should return an Atom object representing the mate.
            - The mate_with method should be defined in the Atom class and should take an Atom object representing the mate as a parameter.
        """
        if self.reproduction_cooldown == 0:
            mate = self.find_mate(atoms)
            if mate:
                self.mate_with(mate, atoms)
                self.reproduction_cooldown = REPRODUCTION_COOLDOWN
                mate.reproduction_cooldown = REPRODUCTION_COOLDOWN

    # EAT
    # Eat nearby food atoms
    def eat(self, atoms):
        """
        Eat nearby food atoms to regain hunger and energy.

        Args:
            atoms (list): A list of Atom objects representing the population.

        Returns:
            None

        This function iterates over the atoms list and checks if the current atom is a food atom
        and if there is a collision with the current atom.
        If both conditions are met, the atom's hunger and energy are increased by 20
        and the food atom is removed from the atoms list.

        The function exits the loop after finding the first food atom.
        """
        for atom in atoms:
            # Check if the current atom is a food atom and if there is a collision
            if atom.species == "food" and self.check_collision(atom):
                # Increase hunger and energy by 20 and remove the food atom from the list
                self.hunger = min(100, self.hunger + 20)
                self.energy = min(100, self.energy + 20)
                atoms.remove(atom)
                break

    # CHECK BOUNDS
    # Check if the atom is in bounds
    def check_bounds(self):
        """
        Ensure the atom stays within the window bounds.

        This function checks if the atom's position is outside the window bounds.
        If it is, it adjusts the atom's velocity to bounce it back into the window.

        Note:
            - The window bounds are defined by the WINDOW_SIZE constant.
        """
        # Check if the atom's x position is outside the window bounds
        if self.x < 0 or self.x > WINDOW_SIZE:
            # Adjust the x velocity to bounce the atom back into the window
            self.vx *= -1

        # Check if the atom's y position is outside the window bounds
        if self.y < 0 or self.y > WINDOW_SIZE:
            # Adjust the y velocity to bounce the atom back into the window
            self.vy *= -1

        # Ensure the atom's position stays within the window bounds
        self.x = max(0, min(self.x, WINDOW_SIZE))
        self.y = max(0, min(self.y, WINDOW_SIZE))

    # UPDATE POSITION
    # Update the position of the atom
    def update_position(self) -> None:
        """
        Update the position of the atom.

        This function updates the position of the atom based on its velocity.
        It also applies damping to the velocities, ensures the atom stays within the window bounds,
        limits the velocity, updates the energy based on kinetic energy, and updates the trail.

        Returns:
            None
        """
        # Update the position based on the velocity
        self.x += self.vx * SIMULATION_SPEED
        self.y += self.vy * SIMULATION_SPEED

        # Apply damping to velocities
        self.vx *= DAMPING_FACTOR
        self.vy *= DAMPING_FACTOR

        # Ensure the atom stays within the window bounds
        self.check_bounds()

        # Limit the velocity
        self.vx = max(-MAX_VELOCITY, min(self.vx, MAX_VELOCITY))
        self.vy = max(-MAX_VELOCITY, min(self.vy, MAX_VELOCITY))

        # Update energy based on kinetic energy
        self.energy = 0.5 * self.mass * (self.vx**2 + self.vy**2)

        # Update trail
        self.trail.append((self.x, self.y))
        if len(self.trail) > TRAIL_LENGTH:
            self.trail.pop(0)

    # APPLY FORCE
    # Apply a force to the atom
    def apply_force(self, fx: float, fy: float) -> None:
        """
        Apply a force to the atom.

        Args:
            fx (float): The x-component of the force.
            fy (float): The y-component of the force.

        Updates the force components and the velocity of the atom based on the given force.

        The velocity is updated by adding the force divided by the mass of the atom.

        Returns:
            None
        """
        # Update the force components
        self.fx += fx
        self.fy += fy

        # Update the velocity based on the force
        self.vx += fx / self.mass
        self.vy += fy / self.mass

    # RESET FORCE
    # Reset the force components of the atom
    def reset_force(self) -> None:
        """
        Reset the force components of the atom.

        This function sets the x and y components of the force to zero.

        Returns:
            None
        """
        # Reset the x-component of the force
        self.fx = 0.0

        # Reset the y-component of the force
        self.fy = 0.0

    # CHECK COLLISION
    # Check if the atom is colliding with another atom
    def check_collision(self, other):
        """
        Check if the atom is colliding with another atom.

        Args:
            other (Atom): The other atom to check collision with.

        Returns:
            bool: True if the atoms are colliding, False otherwise.
        """
        # Calculate the distance between the atoms
        distance = math.hypot(self.x - other.x, self.y - other.y)

        # Check if the distance is less than the sum of the radii of the atoms
        return distance < self.size + other.size

    # RESOLVE COLLISION
    # Resolve a collision between two atoms
    def resolve_collision(self, other):
        """
        Resolves the collision between two atoms.

        Args:
            other (Atom): The other atom to resolve collision with.

        Returns:
            None
        """
        # Calculate the normal vector
        nx = other.x - self.x  # Calculate the x-component of the normal vector
        ny = other.y - self.y  # Calculate the y-component of the normal vector

        distance = math.hypot(nx, ny)  # Calculate the distance between the atoms

        if distance == 0:  # If the distance is zero, no collision
            return

        nx /= distance  # Normalize the normal vector
        ny /= distance  # Normalize the normal vector

        # Calculate relative velocity
        dvx = self.vx - other.vx  # Calculate the x-component of the relative velocity
        dvy = self.vy - other.vy  # Calculate the y-component of the relative velocity

        # Calculate relative velocity in terms of the normal direction
        vn = (
            dvx * nx + dvy * ny
        )  # Calculate the dot product of the relative velocity and normal vector

        # If velocities are separating, no collision
        if vn > 0:
            return

        # Calculate impulse scalar
        impulse = (2 * vn) / (self.mass + other.mass)  # Calculate the impulse scalar

        # Apply impulse to the atoms and apply collision damping
        self.vx -= (
            impulse * other.mass * nx * COLLISION_DAMPING
        )  # Apply impulse to self
        self.vy -= (
            impulse * other.mass * ny * COLLISION_DAMPING
        )  # Apply impulse to self

        other.vx += (
            impulse * self.mass * nx * COLLISION_DAMPING
        )  # Apply impulse to other
        other.vy += (
            impulse * self.mass * ny * COLLISION_DAMPING
        )  # Apply impulse to other

    # FLEE
    # Flee from nearby atoms
    def flee(self, atoms):
        """
        Flee from nearby atoms. If the atom is healthy (health < 30), flee in random directions.

        Parameters:
            self: The current atom.
            atoms: List of atoms to flee from.

        Returns:
            None
        """
        # Flee from nearby atoms
        for atom in atoms:
            if self.check_collision(atom):
                self.resolve_collision(atom)
        # If the atom is healthy
        if self.health < 30:
            # Flee in random directions
            self.vx += random.uniform(-1, 1)
            self.vy += random.uniform(-1, 1)

    # FIND NEAREST FOOD
    # Find the closest food atom
    def find_nearest_food(self, atoms):
        """
        Finds the nearest food atom in the given list of atoms.

        Parameters:
            self (object): The current instance of the class.
            atoms (list): A list of atoms.

        Returns:
            Atom or None: The nearest food atom, or None if no food atom is found.
        """
        closest_food = None
        closest_distance = float("inf")
        for atom in atoms:
            if atom.species == "food":
                distance = math.hypot(self.x - atom.x, self.y - atom.y)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_food = atom
        return closest_food

    # FIND MATE
    # Find a suitable mate
    def find_mate(self, atoms):
        """
        Finds a suitable mate for the current atom from the given list of atoms.

        Parameters:
            self (Atom): The current atom.
            atoms (list): A list of atoms.

        Returns:
            Atom or None: The suitable mate atom, or None if no suitable mate is found.
        """
        # Find a suitable mate
        for other in atoms:
            # Check if the other atom is not the current atom
            if (
                other != self  # Check if the other atom is not the current atom
                # Check if the other atom is healthy
                and other.health > 70
                # Check if the other atom can reproduce
                and other.reproduction_cooldown == 0
                # Check if the other atom is of the same species
                and other.species == self.species
            ):
                return other  # Return the suitable mate
        return None

        # FIND MATE
        def find_mate(self, atoms):
            """
            Finds a suitable mate for the current atom from the given list of atoms.

            Parameters:
                atoms (list): A list of atoms.

            Returns:
                Atom or None: The suitable mate atom, or None if no suitable mate is found.
            """
            mate = None
            # Find a suitable mate for the current atom
            for atom in atoms:
                # Check if the other atom is not the current atom
                if atom is not self and atom.species == self.species:
                    # Check the distance to mate atom and mate if it is less than 100
                    distance = math.hypot(atom.x - self.x, atom.y - self.y)
                    # Check if the distance is less than 100
                    if distance < 100:
                        # Set the mate to the other atom
                        mate = atom
                        break
            return mate

    # CHASE
    # Chase a target
    def chase(self, target):
        """
        Update the velocity of the current atom to chase the target atom.

        Parameters:
            target (Atom): The target atom to chase.

        Returns:
            None
        """
        # Calculate the distance between the atoms
        dx = target.x - self.x
        dy = target.y - self.y
        distance = math.hypot(dx, dy)
        # Check if the distance
        if distance > 0:
            self.vx += dx / distance * 0.1
            self.vy += dy / distance * 0.1

    # WANDER
    # Wander around
    def wander(self):
        """
        Update the velocity of the current atom by adding a random value
        between -0.1 and 0.1 to the x and y components of the velocity.

        This function randomly generates a value between -0.1 and 0.1
        and adds it to the x and y components of the velocity of the current atom.
        This creates a wander behavior where the atom moves in a random direction.

        Parameters:
            None

        Returns:
            None
        """
        self.vx += random.uniform(-0.1, 0.1)
        self.vy += random.uniform(-0.1, 0.1)

    # FIND NEAREST ATOM
    # Find the closest atom
    def find_nearest_atom(self, atoms):
        """
        Finds the nearest atom in the given list of atoms.

        Parameters:
            self (object): The current instance of the class.
            atoms (list): A list of atoms.

        Returns:
            Atom or None: The nearest atom, or None if no atom is found.
        """
        closest_atom = None
        closest_distance = float("inf")
        for other in atoms:
            if other != self:
                distance = math.hypot(self.x - other.x, self.y - other.y)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_atom = other
        return closest_atom

    # MATE WITH
    # Combine genetic traits to produce offspring
    def mate_with(self, mate, atoms):
        """
        Combines the genetic traits of two atoms to produce offspring.

        Parameters:
            self (Atom): The first atom.
            mate (Atom): The second atom.
            atoms (list): A list of atoms.

        Returns:
            None

        This function combines the genetic traits of the two atoms to produce offspring.
        It calls the `combine_atoms` function to generate the new atom.
        If the new atom is successfully created, it is appended to the `atoms` list.
        The reproduction cooldown is set to 60 for both the current atom and the mate.

        Note:
            - The `combine_atoms` function should be defined elsewhere.
            - The `atoms` list should be mutable.
        """
        # Combine genetic traits to produce offspring
        new_atom = combine_atoms(self, mate)
        if new_atom:
            atoms.append(new_atom)
        self.reproduction_cooldown = 60  # Cooldown after reproduction
        mate.reproduction_cooldown = 60

    # DECIDE BEHAVIOR
    # Define the behavior of the atom
    def decide_behavior(self, atoms):
        """
        Decide the behavior of the atom based on its current state and the state of the environment.

        Args:
            atoms (list): A list of all the atoms in the environment.

        Returns:
            None

        The behavior of the atom is determined based on its health, hunger, reproduction cooldown,
        and the state of the environment.
        If the atom is healthy, it will either flee from nearby atoms or move randomly.
        If the atom is not in battle, it will move towards other atoms that are closest to it.
        If the atom is hungry, it will find the nearest food and chase it.
        If it is in mating season, it will find a mate, reproduce, and set its reproduction cooldown.
        If none of the above conditions are met, it will wander or chase the nearest atom.

        Note:
            - The `flee`, `chase`, `find_nearest_food`, `find_mate`, `find_nearest_atom`, and `wander` methods
            should be defined elsewhere.
            - The `atoms` list should be mutable.
        """
        # Reproduction cooldown
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1

        # If the atom is healthy
        if self.health < 30:
            # Flee from nearby atoms
            self.flee(atoms)
        else:
            # Random movement
            self.vx += random.uniform(-0.1, 0.1)
            self.vy += random.uniform(-0.1, 0.1)

        # Behavior: move towards other atoms if not in battle
        if not self.in_battle:
            closest_atom = None
            closest_distance = float("inf")
            for other in atoms:
                if other != self:
                    distance = math.hypot(self.x - other.x, self.y - other.y)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_atom = other
            if closest_atom:
                dx = closest_atom.x - self.x
                dy = closest_atom.y - self.y
                distance = math.hypot(dx, dy)
                if distance > 0:
                    self.vx += dx / distance * 0.1
                    self.vy += dy / distance * 0.1

        # If the atom is hungry, find food
        if self.hunger < 50:
            # Find nearest food
            food = self.find_nearest_food(atoms)
            # If food is found, chase it
            if food:
                self.chase(food)
        elif (
            # Mating season
            MATING_SEASON
            # Reproduction cooldown
            and self.reproduction_cooldown == 0
            # If the atom is healthy and hungry
            and self.hunger > 50
            and self.health > 50
        ):
            # Find nearest mate
            mate = self.find_mate(atoms)
            # If mate is found, chase it
            if mate:
                # Chase mate
                self.chase(mate)
        else:
            # Wander
            nearest_atom = self.find_nearest_atom(atoms)
            # If nearest atom is found, chase it
            if nearest_atom and nearest_atom.health < self.health:
                # Chase nearest atom
                self.chase(nearest_atom)
            else:
                # Wander
                self.wander()


# EVOLVING STRUCTURE
class EvolvingStructure:
    def __init__(self, x, y, size, growth_rate):
        """
        Initializes an instance of the EvolvingStructure class.

        Args:
            x (int or float): The x-coordinate of the instance.
            y (int or float): The y-coordinate of the instance.
            size (int or float): The size of the instance.
            growth_rate (int or float): The rate at which the instance grows per second.

        Returns:
            None
        """
        self.x = x
        self.y = y
        self.size = size  # Initial size
        self.growth_rate = growth_rate  # Increase in size per second
        self.atoms = []  # List of atoms
        self.age = 0  # Age of the structure

    # ADD ATOM
    def add_atom(self, atom):
        """
        Adds an atom to the structure.

        Parameters:
            atom (Atom): The atom to be added.

        Returns:
            None
        """
        # Add atom to structure
        self.atoms.append(atom)

    # GROW
    def grow(self):
        """
        Increases the size of the structure and its atoms,
        updates the age of the structure,
        and slows down the growth rate over time.

        This method increases the size of the structure by adding the growth rate to the current size.
        It then iterates over each atom in the structure and increases its size by the growth rate.
        The age of the structure is incremented by 1.
        Finally, the growth rate is multiplied by 0.99 to slow down the growth over time.

        Parameters:
            self (EvolvingStructure): The instance of the EvolvingStructure class.

        Returns:
            None
        """
        # Increase size
        self.size += self.growth_rate
        # For each atom
        for atom in self.atoms:
            # Increase atom size
            atom.size += self.growth_rate
        # Increase atom age
        self.age += 1
        # Slow down growth over time
        self.growth_rate *= 0.99

    # DRAW
    def draw(self, surface):
        """
        Draws the structure and its atoms on the given surface.

        Args:
            surface (pygame.Surface): The surface to draw on.

        Returns:
            None
        """
        # For each atom
        for atom in self.atoms:
            # Draw atom
            draw_atom(surface, atom)
        # Draw circle
        pygame.draw.circle(
            surface, (0, 255, 0), (int(self.x), int(self.y)), int(self.size), 1
        )


# FUNCTIONS FOR EVOLVING STRUCTURE


# CREATE ATOM
def create_atom(atom_type: str, atoms: list[Atom]) -> Atom:
    """
    Creates an atom based on the given atom_type and initializes its properties accordingly.

    Args:
        atom_type (str): The type of atom to create.
        atoms (List[Atom]): The list of atoms to which the new atom will be added.

    Returns:
        Atom: The newly created atom with the specified properties.
    """
    atom = None
    if atom_type == "red":
        atom = Atom(
            # Random position for blue atom in the x direction
            random.randint(0, WINDOW_SIZE),
            # Random position for blue atom in the y direction
            random.randint(0, WINDOW_SIZE),
            random.uniform(-SPEED, SPEED),
            random.uniform(-SPEED, SPEED),
            1.0,  # Mass for red atoms
            1.0,  # Charge for red atoms
            ATOM_SIZE,
            # Red color
            ATOM_COLORS[1],
            atom_type,
        )

    elif atom_type == "blue":
        atom = Atom(
            # Random position for blue atom in the x direction
            random.randint(0, WINDOW_SIZE),
            # Random position for blue atom in the y direction
            random.randint(0, WINDOW_SIZE),
            # Random initial velocity for blue atoms
            random.uniform(-SPEED, SPEED),
            # Random initial velocity for blue atoms
            random.uniform(-SPEED, SPEED),
            1.0,  # Mass for blue atoms
            -1.0,  # Charge for blue atoms
            ATOM_SIZE,
            # Blue color
            ATOM_COLORS[3],
            atom_type,
        )

    elif atom_type == "food":
        atom = Atom(
            # Random position for food in the x direction
            random.randint(0, WINDOW_SIZE),
            # Random position for food in the y direction
            random.randint(0, WINDOW_SIZE),
            0.0,  # No initial velocity for food
            0.0,  # No initial velocity for food
            0.1,  # Small mass for food atoms to avoid division by zero
            0.0,  # No charge for food atoms
            FOOD_SIZE,
            FOOD_COLORS[1],
            atom_type,
        )

    return atom  # Return the created atom


# APPLY GRAVITY AND FORCES
# https://en.wikipedia.org/wiki/Net_force
def apply_gravity_and_forces(atoms: list[Atom]) -> None:
    """Apply gravity and forces to atoms."""
    for atom in atoms:
        atom.reset_force()  # Reset the force components of the atom

    # For each pair of atoms
    for i, a in enumerate(atoms):
        for j, b in enumerate(atoms[i + 1 :], start=i + 1):
            dx = b.x - a.x  # Distance in the x direction
            dy = b.y - a.y  # Distance in the y direction
            distance = math.hypot(dx, dy)  # Distance between the atoms

            # Check if the distance is less than the sum of the radii of the atoms
            if distance == 0 or distance > FORCE_THRESHOLD:
                continue

            # Gravitational force
            Fg = GRAVITY * a.mass * b.mass / distance**2
            # Coulomb (electric) force
            if a.charge == 0 or b.charge == 0:
                Fe = 0
            else:
                Fe = COULOMB * a.charge * b.charge / distance**2

            # Net force components
            Fx = (Fg - Fe) * dx / distance
            Fy = (Fg - Fe) * dy / distance

            # Scale the forces
            Fx *= FORCE_SCALE
            Fy *= FORCE_SCALE

            # Apply the force to the atoms
            a.apply_force(Fx, Fy)
            b.apply_force(-Fx, -Fy)

    # Apply forces to atoms
    for atom in atoms:
        atom.apply_force(-atom.vx, -atom.vy)

    # Wandering and Reproducing
    for atom in atoms:
        atom.wander()
        atom.check_bounds()
        atom.eat(atoms)
        atom.reproduce(atoms)
        atom.find_nearest_food(atoms)
        # Checking collisions with other atoms
        for other in atoms:
            if other != atom and atom.check_collision(other):
                atom.resolve_collision(other)
        atom.flee(atoms)
        atom.update_position()


# Handle Collisions
def handle_collisions(atoms):
    """
    Handle collisions between atoms in the simulation.

    Parameters:
        atoms (list): A list of Atom objects representing the atoms in the simulation.

    Returns:
    """
    # Check for collisions between atoms
    # For each atom
    for i in range(len(atoms)):
        # For each other atom
        for j in range(i + 1, len(atoms)):
            # If atoms collide
            if atoms[i].check_collision(atoms[j]):
                atoms[i].resolve_collision(atoms[j])


# Draw Atom with thickness based on zoom level
def draw_atom(surface, atom):
    """
    Draw an atom on the given surface with trails and a health bar.

    Args:
        surface: The surface to draw on.
        atom: The atom object to be drawn.

    Returns:
        None
    """
    # Calculate color intensity based on energy
    intensity = min(255, int(atom.energy * 10))  # Adjust the scaling factor as needed
    color = (
        (intensity, intensity, 255) if atom.charge > 0 else (255, intensity, intensity)
    )
    # Draw atom with thickness based on zoom level
    pos_x = int((atom.x - PAN_X) * ZOOM_LEVEL)
    pos_y = int((atom.y - PAN_Y) * ZOOM_LEVEL)
    pygame.draw.circle(surface, color, (pos_x, pos_y), int(atom.size * ZOOM_LEVEL))
    # Draw trails with thickness based on zoom level
    trail_thickness = max(
        1, int(math.exp(min(ZOOM_LEVEL - 1, 3)))
    )  # Exponential adjustment, limit to prevent overflow
    for i in range(len(atom.trail) - 1):
        # Calculate trail coordinates
        trail_x1 = int((atom.trail[i][0] - PAN_X) * ZOOM_LEVEL)
        trail_y1 = int((atom.trail[i][1] - PAN_Y) * ZOOM_LEVEL)
        trail_x2 = int((atom.trail[i + 1][0] - PAN_X) * ZOOM_LEVEL)
        trail_y2 = int((atom.trail[i + 1][1] - PAN_Y) * ZOOM_LEVEL)
        # Draw a line with trail thickness
        pygame.draw.line(
            surface, color, (trail_x1, trail_y1), (trail_x2, trail_y2), trail_thickness
        )
    # Draw health bar
    health_bar_length = int(ATOM_SIZE * 2 * ZOOM_LEVEL)
    # Draw health bar height
    health_bar_height = int(2 * ZOOM_LEVEL)
    # Adjust the health bar position to keep it centered
    health_bar_x = pos_x - health_bar_length // 2
    health_bar_y = pos_y - int(atom.size * ZOOM_LEVEL) - health_bar_height - 2
    # Draw health bar
    pygame.draw.rect(
        surface,
        (255, 0, 0),
        (health_bar_x, health_bar_y, health_bar_length, health_bar_height),
    )
    pygame.draw.rect(
        surface,
        (0, 255, 0),
        (
            health_bar_x,
            health_bar_y,
            int(health_bar_length * (atom.health / 100)),
            health_bar_height,
        ),
    )


# Save Simulation
def save_simulation(atoms, filename="simulation.pkl"):
    """
    Save the simulation data to a file.

    Args:
        atoms (list): A list of atoms representing the simulation state.
        filename (str, optional): The name of the file to save the simulation data. Defaults to "simulation.pkl".

    Returns:
        None
    """
    if "../" in filename or "..\\" in filename:
        raise Exception("Invalid file path")
    with open(filename, "wb") as file:
        pickle.dump(atoms, file)


# Load Simulation
def load_simulation(filename="simulation.pkl"):
    """
    Load the simulation data from a file.

    Args:
        filename (str, optional): The name of the file to load the simulation data from.
            Defaults to "simulation.pkl".

    Returns:
        list: The simulation data loaded from the file.
    """
    if "../" in filename or "..\\" in filename:
        raise Exception("Invalid file path") 
    with open(filename, "rb") as file:
        return pickle.load(file)


# Zoom At mouse position
def zoom_at(mouse_pos, zoom_factor):
    """
    Updates the zoom level based on the provided zoom factor.

    Args:
        mouse_pos (tuple): The current mouse position.
        zoom_factor (float): The factor by which to zoom.

    Returns:
        None
    """
    global ZOOM_LEVEL, PAN_X, PAN_Y
    old_zoom_level = ZOOM_LEVEL
    ZOOM_LEVEL *= zoom_factor
    # Convert mouse position to world coordinates
    mouse_x, mouse_y = mouse_pos
    # Calculate new and old world coordinates
    world_x = (mouse_x / old_zoom_level) + PAN_X
    world_y = (mouse_y / old_zoom_level) + PAN_Y
    new_world_x = (mouse_x / ZOOM_LEVEL) + PAN_X
    new_world_y = (mouse_y / ZOOM_LEVEL) + PAN_Y
    # Update PAN_X and PAN_Y
    PAN_X += world_x - new_world_x
    PAN_Y += world_y - new_world_y


# Combine Atoms into new Atom
def combine_atoms(atom1, atom2):
    # Logic to combine atoms into a more complex structure
    new_mass = atom1.mass + atom2.mass
    new_charge = (atom1.charge + atom2.charge) / 2
    new_size = (atom1.size + atom2.size) / 2
    new_color = (
        (atom1.color[0] + atom2.color[0]) // 2,
        (atom1.color[1] + atom2.color[1]) // 2,
        (atom1.color[2] + atom2.color[2]) // 2,
    )
    # Combine health values
    new_health = min(atom1.health, atom2.health)
    new_atom = Atom(
        atom1.x,
        atom1.y,
        (atom1.vx + atom2.vx) / 2,
        (atom1.vy + atom2.vy) / 2,
        new_mass,
        new_charge,
        new_size,
        new_color,
        atom1.species,
    )
    # Update health
    new_atom.health = new_health
    return new_atom


# Update Atoms and Structures
def update_atoms(atoms, structures):
    """
    Update the positions and attributes of atoms in the simulation.

    Parameters:
        atoms (list): A list of Atom objects representing the atoms in the simulation.
        structures (list): A list of Structure objects representing the structures in the simulation.

    Returns:
        None
    """
    # Update atom positions
    for atom in atoms:
        # Update age
        atom.age += 1
        # Update hunger and energy
        atom.hunger -= 0.1
        atom.energy -= 0.2
        # Check if the atom is dead
        if atom.hunger <= 0 or atom.energy <= 0:
            atom.health -= 0.5
        if atom.health <= 0:
            atoms.remove(atom)
        # AI decision-making
        atom.decide_behavior(atoms)


# Update Structures
def update_structures(structures):
    """
    Updates all structures in the given list by advancing their growth rate.

    Args:
        structures (list): A list of structure objects to be updated.

    Returns:
        None
    """
    # Update structure positions
    for structure in structures:
        # Update structure growth rate
        structure.grow()


# Constants for the number of structures
NUM_STRUCTURES = 5  # Define the number of structures

# Initialize pygame
pygame.init()

# Create window
window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))

# Set window caption
pygame.display.set_caption("Life Simulation by L4ndbo (version: 6_21062024)")

# Create atoms
atoms = []
for _ in range(NUM_ATOMS):
    species = random.choice(["red", "blue"])
    atoms.append(create_atom(species, atoms))

# Create food atoms
for _ in range(NUM_FOOD_ATOMS):
    atoms.append(create_atom("food", atoms))

# Create structures
structures = []

# Add a new structure for demonstration
structure = EvolvingStructure(400, 400, 20, 0.1)
structures.append(structure)

# Create additional structures
for _ in range(NUM_STRUCTURES):
    structures.append(
        EvolvingStructure(
            random.randint(0, WINDOW_SIZE), random.randint(0, WINDOW_SIZE), 20, 0.1
        )
    )

# Set simulation speed
clock = pygame.time.Clock()

# Set pan speed
panning = False

# Set the pan start position
pan_start_x, pan_start_y = 0, 0
selected_atom = None

# Start time of the simulation
start_time = pygame.time.get_ticks()

# running simulation
running = True
while running:
    # Poll for events
    # Url: https://www.pygame.org/docs/ref/event.html

    # Set simulation speed
    clock.tick(SIMULATION_SPEED)

    # Fill window with black
    window.fill((0, 0, 0))  # Fill window with black
    for event in pygame.event.get():
        # Check for quit event
        # pygame.QUIT event means the user clicked X to close your window
        if event.type == pygame.QUIT:
            running = False

        # Check for keyboard events (KEYDOWN for key pressed, KEYUP for key released)
        elif event.type == pygame.KEYDOWN:
            # Check for key presses

            # P for PAUSE SIMULATION
            if event.key == pygame.K_p:  # Pause
                SIMULATION_SPEED = 0 if SIMULATION_SPEED != 0 else 1

            # R for RESET SIMULATION
            elif event.key == pygame.K_r:  # Reset
                atoms = []
                structures = []

                # Create atoms
                for _ in range(NUM_ATOMS):
                    species = random.choice(["red", "blue"])
                    atoms.append(create_atom(species, atoms))

                # Create food atoms
                for _ in range(NUM_FOOD_ATOMS):
                    atoms.append(create_atom("food", atoms))

                # Create additional structures
                for _ in range(NUM_STRUCTURES):
                    structures.append(
                        EvolvingStructure(
                            random.randint(0, WINDOW_SIZE),
                            random.randint(0, WINDOW_SIZE),
                            20,
                            0.1,
                        )
                    )
                start_time = pygame.time.get_ticks()

            # TIME OF THE SIMULATION CONTROLS
            # + for FAST FORWARD
            elif event.key == pygame.K_PLUS:  # Fast forward
                SIMULATION_SPEED = 2

            # - for SLOW MOTION
            elif event.key == pygame.K_MINUS:  # Slow motion
                SIMULATION_SPEED = 0.5

            # SIMULATION STATE CONTROLS
            # L for LOAD SIMULATION
            elif event.key == pygame.K_l:  # Load simulation
                atoms = load_simulation()
            # S for SAVE SIMULATION
            elif event.key == pygame.K_s:  # Save simulation
                save_simulation(atoms)

            # N for NEW SIMULATION
            elif event.key == pygame.K_n:  # New simulation
                atoms = []
                structures = []
                start_time = pygame.time.get_ticks()

            # Q for QUIT
            elif event.key == pygame.K_q:  # Quit
                running = False

        # Check for mouse events
        # Url: https://www.pygame.org/docs/ref/mouse.html

        # A mouse button is pressed
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mx, my = event.pos
                for atom in atoms:
                    pos_x = int((atom.x - PAN_X) * ZOOM_LEVEL)  # Convert to integer
                    pos_y = int((atom.y - PAN_Y) * ZOOM_LEVEL)  # Convert to integer
                    # Check if the mouse is within the radius of the atom
                    if math.hypot(mx - pos_x, my - pos_y) <= atom.size * ZOOM_LEVEL:
                        selected_atom = atom
                        break
            elif event.button == 3:  # Right click
                panning = True
                (
                    pan_start_x,
                    pan_start_y,
                ) = event.pos  # Save the starting position of the mouse
            elif event.button == 4:  # Scroll up
                zoom_at(event.pos, 1.1)
            elif event.button == 5:  # Scroll down
                zoom_at(event.pos, 1 / 1.1)

        # A mouse button is released
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3:  # Right click
                panning = False

        # A mouse movement has occurred
        elif event.type == pygame.MOUSEMOTION:  # Mouse movement
            if panning:
                dx, dy = event.rel
                PAN_X -= dx / ZOOM_LEVEL
                PAN_Y -= dy / ZOOM_LEVEL

    # Apply gravity and forces to atoms
    apply_gravity_and_forces(atoms)

    # Handle collisions with other atoms
    handle_collisions(atoms)

    # Update atoms and structures
    update_atoms(atoms, structures)
    update_structures(structures)

    # Draw atoms
    for atom in atoms:
        atom.decide_behavior(atoms)  # AI decision making
        atom.update_position()  # Draw atoms using the draw_atom function
        draw_atom(window, atom)  # Draw atoms with thickness based on zoom level

    # Draw food
    for atom in atoms:
        if atom.species == "food":
            draw_atom(window, atom)

    # Draw structures with thickness based on zoom level
    for structure in structures:
        structure.grow()  # Increase structure size
        structure.draw(window)  # Draw structure with thickness based on zoom level

    # Calculate total simulation time
    current_time = pygame.time.get_ticks()
    total_time = (current_time - start_time) / 1000  # Convert to seconds

    # If an atom is selected, update the window caption
    if selected_atom:
        pygame.display.set_caption(
            f"Life Simulation by L4ndbo (version: 6_21062024) || Selected Atom: {selected_atom.species} | Age: {selected_atom.age} | Health: {selected_atom.health} | Hunger: {selected_atom.hunger} | Energy: {selected_atom.energy}"
        )
    else:
        # Calculate the number of red atoms
        red_count = sum(1 for atom in atoms if atom.species == "red")
        # Calculate the number of blue atoms
        blue_count = sum(1 for atom in atoms if atom.species == "blue")
        # Calculate the total age of the atoms
        total_age = sum(atom.age for atom in atoms)
        # Calculate the average age of the atoms
        average_age = total_age / len(atoms) if atoms else 0
        # Calculate the oldest age of the atoms
        oldest_age = max(atom.age for atom in atoms) if atoms else 0
        # Calculate the number of structures
        structure_count = len(structures)

        # Update window caption
        pygame.display.set_caption(
            f"Life Simulation by L4ndbo (version: 6_21062024) || Toal Simulation Time: {total_time:.2f} | Total Red: {red_count} | Total Blue: {blue_count} | Avg Atom Age: {average_age:.2f} | Oldest Atom: {oldest_age} | Total Structures: {structure_count}"
        )

    # Update simulation speed
    pygame.display.flip()
    clock.tick(60)  # Limit to 60 frames per second

# Quit Pygame
pygame.quit()
# Exit script
exit()
