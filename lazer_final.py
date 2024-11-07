
from itertools import combinations
from PIL import Image, ImageDraw, ImageFilter  
import time 

class Input:
    """
    The Input class processes a `.bff` file to extract configuration data for a Lazor puzzle.

    Notation for blocks:
    #   x = no block allowed
    #   o = blocks allowed
    #   A = fixed reflect block
    #   B = fixed opaque block
    #   C = fixed refract block
    """
    def __init__(self, filename: str):
        """
        Initializes the Input class with the provided `.bff` filename.

        Args:
            filename (str): The name of the `.bff` file to be processed.

        Raises:
            SystemExit: If the file does not have a `.bff` extension.
        """
        # Check if the provided file has a `.bff` extension; exit if not
        if not filename.lower().endswith('.bff'):
            raise SystemExit("Invalid file type. Please provide a `.bff` file.")
        
        # Initialize file name, grid width, and height
        self.filename = filename
        self.width = 0
        self.height = 0

    def __call__(self) -> dict:
        """
        Processes the `.bff` file to extract the Lazor puzzle configuration, including grid layout, block positions, lazor paths, and target points.

        Returns:
            dict: A dictionary containing:
                - "Size" (list): Width and height of the grid.
                - "o_l" (list): Open positions where blocks can be placed.
                - "Lazers" (list): Lazor start points and directions.
                - "Points" (list): Points of interest (target points).
                - "A" (int): Count of type A blocks (reflect).
                - "B" (int): Count of type B blocks (opaque).
                - "C" (int): Count of type C blocks (refract).
                - "x_l" (list): Positions where blocks cannot be placed.
                - "A_l" (list): Fixed positions of type A blocks.
                - "B_l" (list): Fixed positions of type B blocks.
                - "C_l" (list): Fixed positions of type C blocks.

        Raises:
            SystemExit: If essential markers or data (e.g., lazors, blocks) are missing or improperly formatted.
        """
        # Initialize counts for each block type and empty lists for positions
        A_count, B_count, C_count = 0, 0, 0
        o_positions, x_positions = [], []
        A_blocks, B_blocks, C_blocks = [], [], []  # Lists to hold positions of blocks A, B, and C
        lazors, points = [], []  # Lists for laser start points and target points

        # Open and read the `.bff` file line by line
        try:
            with open(self.filename, 'r') as file:
                lines = file.read().splitlines()
        except FileNotFoundError:
            raise SystemExit(f"File `{self.filename}` not found.")

        # Locate the grid start and stop markers in the file
        try:
            start_idx = lines.index("GRID START") + 1
            stop_idx = lines.index("GRID STOP")
        except ValueError:
            raise SystemExit("GRID START or GRID STOP not found in the test file.")

        # Parse the grid data to identify open, blocked, and specific block positions
        for y, line in enumerate(lines[start_idx:stop_idx], start=1):
            tokens = line.split()
            for x, token in enumerate(tokens):
                position = [x, y]
                if token == 'o':  # Open position
                    o_positions.append(position)
                elif token == 'x':  # Blocked position
                    x_positions.append(position)
                elif token == 'A':  # Block A position
                    A_blocks.append(position)
                elif token == 'B':  # Block B position
                    B_blocks.append(position)
                elif token == 'C':  # Block C position
                    C_blocks.append(position)
            # Track the maximum width and current height of the grid
            self.height = y
            self.width = max(self.width, len(tokens))

        # Check if there are open positions for placing blocks
        if not o_positions:
            raise SystemExit("No open positions ('o') found in the grid.")

        # Adjust the vertical coordinates to match the internal grid representation
        for block_list in [o_positions, x_positions, A_blocks, B_blocks, C_blocks]:
            for position in block_list:
                position[1] = self.height - position[1]

        # Parse the file for block counts, lasers, and points of interest (POI)
        for line in lines[stop_idx + 1:]:
            line = line.strip()
            if not line or line.startswith('#'):  # Ignore comments and empty lines
                continue

            tokens = line.split()
            identifier = tokens[0]

            # Block counts for A, B, and C
            if identifier in {'A', 'B', 'C'}:
                try:
                    count = int(tokens[-1])
                except (IndexError, ValueError):
                    raise SystemExit(f"Invalid count for block '{identifier}'.")
                if identifier == 'A':
                    A_count = count
                elif identifier == 'B':
                    B_count = count
                elif identifier == 'C':
                    C_count = count
            # Lazor configuration: starting position and direction
            elif identifier == 'L':
                try:
                    lazors.append([int(tok) for tok in tokens[1:5]])
                except (IndexError, ValueError):
                    raise SystemExit("Invalid Lazor configuration.")
            # Point of interest coordinates
            elif identifier == 'P':
                try:
                    points.append([int(tok) for tok in tokens[1:3]])
                except (IndexError, ValueError):
                    raise SystemExit("Invalid POI coordinates.")

        # Ensure there are blocks and lazors specified for the puzzle
        if not any([A_count, B_count, C_count]):
            raise SystemExit("No blocks specified to solve the Lazor puzzle.")
        if not lazors:
            raise SystemExit("No Lazors provided in the test file.")

        # Adjust coordinates for lazors and points to fit the grid representation
        for lazor in lazors:
            lazor[0] *= 0.5
            lazor[1] = self.height - (lazor[1] * 0.5)
            lazor[2] *= 0.5
            lazor[3] *= -0.5

        for poi in points:
            poi[0] *= 0.5
            poi[1] = self.height - (poi[1] * 0.5)

        # Organize all extracted data into a dictionary for easy access
        input_data = {
            "Size": [self.width, self.height],
            "o_l": o_positions,
            "Lazers": lazors,
            "Points": points,
            "A": A_count,
            "B": B_count,
            "C": C_count,
            "x_l": x_positions,
            "A_l": A_blocks,
            "B_l": B_blocks,
            "C_l": C_blocks
        }

        return input_data  # Return the parsed configuration data



class Lazor_Solution:
    '''
    This class processes grid and lazor data from input, finds possible 
    block combinations, and checks if any configuration meets the criteria.
    Also includes path tracking for further visualization.
    '''

    def __init__(self, input_data):
        # Initialize grid data and configuration for the Lazor puzzle
        self.o_l = input_data['o_l']  # Open positions
        self.size = input_data['Size']  # Size of the grid
        self.lazers = input_data['Lazers']  # Initial laser positions and directions
        self.points = input_data['Points']  # Target points to hit with lasers
        self.A = input_data['A']  # Number of block type A
        self.B = input_data['B']  # Number of block type B
        self.C = input_data['C']  # Number of block type C
        self.A_l = input_data['A_l']  # Fixed positions of block type A
        self.B_l = input_data['B_l']  # Fixed positions of block type B
        self.C_l = input_data['C_l']  # Fixed positions of block type C
        self.block_layout = {}  # Dictionary to store the layout of blocks on the grid
        self.solution_paths = []  # Stores successful laser paths for visualization

    def __call__(self):
        """
        Generates possible block configurations on the grid and evaluates them to find a solution that meets the puzzle criteria.

        Returns:
            None
        """
        # Generate all possible combinations of open positions for blocks of type A
        o_lA = list(combinations(self.o_l, self.A))
        for i_a in o_lA:
            o_l = [pos for pos in self.o_l if pos not in i_a]
            a_comb = list(i_a)
            
            # Generate combinations of remaining positions for blocks of type B
            o_lB = list(combinations(o_l, self.B))
            for i_b in o_lB:
                o_l_b = [pos for pos in o_l if pos not in i_b]
                b_comb = list(i_b)
                
                # Generate combinations of remaining positions for blocks of type C
                o_lC = list(combinations(o_l_b, self.C))
                for i_c in o_lC:
                    c_comb = list(i_c)
                    
                    # Create block layout with the chosen positions for A, B, and C blocks
                    self.block_layout = {}
                    for j, block_position in enumerate([a_comb, b_comb, c_comb]):
                        block_name = ['A', 'B', 'C'][j]
                        for pos in block_position:
                            self.block_layout[(pos[0], pos[1])] = block_name

                    # Add predefined fixed block positions to the layout
                    for block_list, label in zip([self.A_l, self.B_l, self.C_l], ['A', 'B', 'C']):
                        for pos in block_list:
                            self.block_layout[(pos[0], pos[1])] = label

                    # Check if the current configuration solves the puzzle
                    if self.check_solution():
                        return self.block_layout, self.solution_paths

    def check_solution(self) -> bool:
        """
        Checks if the current block configuration solves the Lazor puzzle by tracing each laser path and verifying if all target points are hit.

        Returns:
            bool: True if the configuration satisfies the puzzle requirements by hitting all target points, False otherwise.
        """
        # Reset the solution paths for a new attempt
        self.solution_paths = []
        remaining_points = list(self.points)  # Points that need to be hit by lasers
        active_lasers = list(self.lazers)  # List of currently active lasers
        all_points_hit = False  # Track if all points have been hit
        
        # Process each laser until no active lasers are left
        while active_lasers:
            x, y, vx, vy = active_lasers.pop(0)
            current_path = [(x, y)]  # Start tracking this laser's path
            
            while True:
                pos = [x, y]
                
                # Check if this position hits any remaining target points
                if pos in remaining_points:
                    remaining_points.remove(pos)
                    if not remaining_points and not all_points_hit:
                        all_points_hit = True  # Mark that all points are hit but continue tracing the path
                
                # Determine the next position based on laser direction
                if x.is_integer() and vx < 0:
                    upd_pos = (x - 1, (y * 2 - 1) / 2)
                elif x.is_integer() and vx > 0:
                    upd_pos = (x, (y * 2 - 1) / 2)
                elif not x.is_integer() and vy < 0:
                    upd_pos = ((x * 2 - 1) / 2, y - 1)
                else:
                    upd_pos = ((x * 2 - 1) / 2, y)

                # If the laser is outside the grid, save the path and stop
                if not self.pos_chk(upd_pos):
                    self.solution_paths.append(current_path)
                    break

                # Check if the laser hits a block
                if upd_pos in self.block_layout:
                    block_type = self.block_layout[upd_pos]
                    if block_type == 'A':
                        # Reflect laser if it hits a block A
                        x, y, vx, vy = self.reflect((x, y, vx, vy))
                        current_path.append((x, y))  # Add reflection point to path
                    elif block_type == 'C':
                        # Refract laser if it hits a block C
                        paths = self.refract((x, y, vx, vy))
                        for new_x, new_y, new_vx, new_vy in paths:
                            active_lasers.append([new_x, new_y, new_vx, new_vy])
                            # Add refracted paths to the solution paths
                            self.solution_paths.append(current_path + [(new_x, new_y)])
                        break
                    else:  
                        # Block B stops the laser
                        self.solution_paths.append(current_path)  # Save path up to block
                        break
                else:
                    # Move the laser to the next position if no block is encountered
                    x += vx
                    y += vy
                    current_path.append((x, y))

        return all_points_hit  # Return True only if all target points were hit

    def pos_chk(self, upd_pos) -> bool:
        """
        Checks if the given position is within the bounds of the grid.

        Args:
            upd_pos (tuple): A tuple (x, y) representing the position to check.

        Returns:
            bool: True if the position is within grid boundaries, False otherwise.
        """

        x, y = upd_pos
        return 0 <= x < self.size[0] and 0 <= y < self.size[1]

    def reflect(self, lazer) -> tuple:
        """
        Calculates the reflection of a laser upon hitting a reflective block.

        Args:
            laser (tuple): A tuple (x, y, vx, vy) representing the laser's current position and velocity.

        Returns:
            tuple: A tuple (new_x, new_y, new_vx, new_vy) representing the reflected position and velocity of the laser.
        """
        
        x, y, vx, vy = lazer
        if x.is_integer():
            return x - vx, y + vy, -vx, vy  # Reflect in x-direction
        else:
            return x + vx, y - vy, vx, -vy  # Reflect in y-direction

    def refract(self, lazer) -> list:
        """
        Calculates the refraction of a laser upon hitting a refractive block, creating two paths: one continues the original path, and one reflects.

        Args:
            laser (tuple): A tuple (x, y, vx, vy) representing the laser's current position and velocity.

        Returns:
            list: A list containing two tuples, each representing a path of the refracted laser.
        """
        x, y, vx, vy = lazer
        path1 = (x + vx, y + vy, vx, vy)  # Original path
        # Create a second path based on the orientation of the position
        if x.is_integer():
            # refract lazer for side direction
            path2 = (x - vx, y + vy, -vx, vy) 
        else:
            # refrac lazer for top and botton side lazer
            path2 = (x + vx, y - vy, vx, -vy)  
        return [path1, path2]  # Return both the original and new paths


class SaveSolution:
    """
    This class handles saving and rendering the Lazor puzzle solution as an image file.
    """
    def __init__(self, filename: str, info: dict, block_layout: dict, solution_paths: list):
        """
        Initializes the SaveSolution class with details required to save and render the Lazor puzzle solution.

        Args:
            filename (str): The name of the file where the solution image will be saved.
            info (dict): A dictionary containing the Lazor puzzle configuration and solution data.
            block_layout (dict): A dictionary mapping positions to block types in the grid.
            solution_paths (list): A list of paths traced by lazors in the solution.
        """
        self.filename = filename  # Output filename
        self.info = info  # Puzzle information (size, laser positions, etc.)
        self.block_layout = block_layout  # Final layout of blocks in the solution
        self.solution_paths = solution_paths  # Laser paths in the solution
        self.lazers = info["Lazers"]  # Initial laser configurations
        self.points = info["Points"]  # Target points to hit with lasers
        self.block_size = 100  # Size of each block in pixels
        self.shadow_offset = 0  # Offset for shadows in block rendering
        self.corner_radius = 20  # Corner radius for rounded blocks

    def __call__(self) -> None:
        """
        Saves the final solution image by building and drawing grid, laser paths, and points.

        Returns:
            None
        """
        if not self.block_layout:
            print("No solution is found for the given file")
            return

        # Retrieve grid size and build a figure based on the solution layout
        size = self.info['Size']
        figure = self.build_figure(size)
        
        # Create an image based on the grid and overlay laser paths and points
        img = self.create_image(size, figure)
        img = self.draw_lazor_paths(img, size)  # Draw laser paths on the image
        self.draw_lazor_points(img, size)  # Draw laser points
        self.save_image(img)  # Save the final solution image

    def build_figure(self, size: list) -> list:
        """
        Creates a grid-based figure representing the Lazor puzzle solution with color codes for each cell based on block types.

        Args:
            size (list): A list containing the width and height of the grid.

        Returns:
            list: A 2D list representing the grid where each cell contains a color code indicating block type or open/blocked status.
        """
        color_mapping = {'A': 1, 'B': 2, 'C': 3, 'x': 4}
        figure = [[0] * size[0] for _ in range(size[1])]  # Initialize grid

        # Assign color codes for each block type in the solution layout
        for (x, y), block_type in self.block_layout.items():
            figure[size[1] - y - 1][x] = color_mapping.get(block_type, 0)
        for x, y in self.info['x_l']:
            figure[size[1] - y - 1][x] = 4  # Mark 'x' (blocked) positions

        return figure

    def create_rounded_rectangle(self, draw: ImageDraw, xy: tuple, radius: int, fill: tuple, outline: tuple = None, width: int = 1) -> None:
        """
        Draws a rounded rectangle on the image with specified dimensions, fill color, outline, and corner radius.

        Args:
            draw (ImageDraw): The ImageDraw object used for drawing.
            xy (tuple): A tuple (x1, y1, x2, y2) representing the bounding box of the rectangle.
            radius (int): The radius of the corners.
            fill (tuple): The fill color of the rectangle.
            outline (tuple, optional): The outline color of the rectangle. Defaults to None.
            width (int, optional): The width of the outline. Defaults to 1.
        """
        x1, y1, x2, y2 = xy
        diameter = 2 * radius
        # Draw rounded corners
        draw.ellipse((x1, y1, x1 + diameter, y1 + diameter), fill=fill, outline=outline)
        draw.ellipse((x2 - diameter, y1, x2, y1 + diameter), fill=fill, outline=outline)
        draw.ellipse((x1, y2 - diameter, x1 + diameter, y2), fill=fill, outline=outline)
        draw.ellipse((x2 - diameter, y2 - diameter, x2, y2), fill=fill, outline=outline)
        # Fill in the middle rectangle sections to complete the rounded shape
        draw.rectangle((x1 + radius, y1, x2 - radius, y2), fill=fill, outline=outline)
        draw.rectangle((x1, y1 + radius, x2, y2 - radius), fill=fill, outline=outline)

    def create_block_with_3d_effect(self, size: int, block_type: str) -> Image:
        # Create an individual block image with 3D styling based on type
        block = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(block)
        
        # Define colors for different block types
        colors = {
            'o': (120, 110, 110),  # Open space color
            'A': (250, 250, 250),  # Light gray for block A
            'B': (28, 28, 28),     # Dark gray for block B
            'C': (197, 212, 238),  # Blue-gray for block C
            'x': (200, 200, 200)   # Blocked space color
        }
        
        base_color = colors.get(block_type, colors['o'])
        padding = 5  # Inner padding for the block
        x1, y1 = padding, padding
        x2, y2 = size - padding, size - padding
        
        if block_type == 'o':
            # Open spaces have a subtle shadow and lighter 3D effect
            shadow_color = tuple(max(0, c - 10) for c in base_color) + (255,)
            self.create_rounded_rectangle(draw, (x1, y1, x2, y2), self.corner_radius, shadow_color)
            inset = 3
            lighter_color = tuple(min(255, c + 5) for c in base_color) + (255,)
            self.create_rounded_rectangle(draw, (x1 + inset, y1 + inset, x2 - inset, y2 - inset), 
                                          self.corner_radius - inset, lighter_color)
            
        elif block_type == 'A':
            # Block A with a highlight effect
            highlight_color = tuple(min(255, c + 10) for c in base_color) + (255,)
            self.create_rounded_rectangle(draw, (x1, y1, x2, y2), self.corner_radius, highlight_color)
            inset = self.shadow_offset
            self.create_rounded_rectangle(draw, (x1 + inset, y1 + inset, x2 - inset, y2 - inset), 
                                          self.corner_radius - inset, base_color + (255,))

        elif block_type == 'C':
            # Block C with shadow and highlight ellipse
            shadow_color = tuple(max(0, c + 10) for c in base_color) + (255,)
            self.create_rounded_rectangle(draw, (x1, y1, x2, y2), self.corner_radius, shadow_color)
            inset = 5
            self.create_rounded_rectangle(draw, (x1 + inset, y1 + inset, x2 - inset, y2 - inset), 
                                          self.corner_radius - inset, base_color + (255,))
            highlight_radius = size // 4
            highlight_pos = (x1 + highlight_radius, y1 + highlight_radius)
            draw.ellipse((highlight_pos[0] - highlight_radius // 2, highlight_pos[1] - highlight_radius // 2,
                          highlight_pos[0] + highlight_radius // 2, highlight_pos[1] + highlight_radius // 2),
                         fill=(212, 223, 243)) 

        elif block_type == 'B':
            # Block B with a dark shadow effect
            shadow_color = tuple(max(0, c + 20) for c in base_color) + (255,)
            self.create_rounded_rectangle(draw, (x1, y1, x2, y2), self.corner_radius, shadow_color)
            inset = self.shadow_offset
            self.create_rounded_rectangle(draw, (x1 + inset, y1 + inset, x2 - inset, y2 - inset), 
                                          self.corner_radius - inset, base_color + (255,))
      
        return block

    def create_image(self, size, figure) -> Image:
        # Initialize the main image with a background color and gradient effect
        img = Image.new("RGBA", (size[0] * self.block_size, size[1] * self.block_size), (100, 93, 93, 255))
        
        # Apply a subtle gradient effect
        gradient = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(gradient)
        for i in range(img.height):
            alpha = int(255 * (1 - i / img.height))
            draw.line([(0, i), (img.width, i)], fill=(255, 255, 255, alpha // 8))
        img = Image.alpha_composite(img, gradient)
        
        # Place each block on the grid
        for y, row in enumerate(figure):
            for x, block_type in enumerate(row):
                block_type_map = {0: 'o', 1: 'A', 2: 'B', 3: 'C', 4: 'x'}
                block = self.create_block_with_3d_effect(self.block_size, block_type_map[block_type])
                img.paste(block, (x * self.block_size, y * self.block_size), block)
                
        return img

    def draw_lazor_paths(self, img: Image, size: list) -> Image:
        """
        Draws laser paths with a glow effect on the image.

        Args:
            img (Image): The image on which to draw laser paths.
            size (list): A list containing the width and height of the grid.

        Returns:
            Image: The updated image with laser paths and glow effect.
        """
        
        glow_layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(glow_layer)
        
        for path in self.solution_paths:
            points = []
            for x, y in path:
                px = x * self.block_size
                py = (size[1] - y) * self.block_size
                points.append((px, py))
                
            # Create glow effect by drawing lines with decreasing width and opacity
            for width, opacity in [(7, 30), (5, 50), (3, 80), (1, 255)]:
                if len(points) > 1:
                    draw.line(points, fill=(255, 0, 0, opacity), width=width)
        
        # Apply Gaussian blur to create a smooth glow effect
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=1))
        return Image.alpha_composite(img, glow_layer)

    def draw_lazor_points(self, img: Image, size: list) -> None:
        """
        Draws laser start points and target points on the image.

        Args:
            img (Image): The image on which to draw laser points.
            size (list): A list containing the width and height of the grid.
        """
        
        draw = ImageDraw.Draw(img)
        colors = {
            'lazor': (255, 0, 0, 255),  # Red for laser start points
            'poi': (0, 0, 0, 255)       # Black for points of interest
        }

        # Draw laser starting points
        for x, y, *_ in self.lazers:
            xp, yp = x * self.block_size, (size[1] - y) * self.block_size
            draw.ellipse([(xp - 5, yp - 5), (xp + 5, yp + 5)], fill=colors['lazor'])

        # Draw target points of interest
        for x, y in self.points:
            xp, yp = x * self.block_size, (size[1] - y) * self.block_size
            draw.ellipse([(xp - 8, yp - 8), (xp + 8, yp + 8)], fill=colors['poi'])

    def save_image(self, img: Image) -> None:
        """
        Saves the final solution image to the specified filename.

        Args:
            img (Image): The image to be saved.
        """
        
        if ".bff" in self.filename:
            self.filename = self.filename.replace(".bff", "")
        if not self.filename.endswith(".png"):
            self.filename += ".png"  # Ensure the file ends with .png
        img.save(self.filename)


    
if __name__ == "__main__":
    filenames = ["yarn_5.bff", "tiny_5.bff", "showstopper_4.bff", "numbered_6.bff",
                 "mad_1.bff", "mad_7.bff", "mad_4.bff", "dark_1.bff"]
    execution_times = []

    for filename in filenames:
        start_time = time.time()

        input_processor = Input(filename)
        input_data = input_processor()

        lazor_solver = Lazor_Solution(input_data)
        solution = lazor_solver()
        
        if solution:
            block_layout, paths = solution  # Unpack the solution tuple
            save_solution = SaveSolution(filename, input_data, block_layout, paths)
            save_solution()
            print(f"Solution found and saved for {filename}")
        else:
            print(f"No solution found for {filename}")

        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)
        print(f"Execution time for {filename}: {execution_time:.2f} seconds")

    print("\nAll execution times:", [f"{t:.2f}" for t in execution_times])