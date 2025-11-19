import os
import random
from PIL import Image
import collections

NUM_OUTPUT_IMAGES = 100000
VARIATIONS_PER_MESSY = 5

INPUT_DIR_NAME = 'tools'
MESSY_DIR_NAME = 'messy'
NEAT_DIR_NAME = 'neat'

MAX_COPIES_PER_OBJECT = 4
MAX_PLACEMENT_ATTEMPTS = 1000000
CANVAS_SIZE = (256, 256)
RULER_SPACING = 3
HORIZONTAL_SPACING = 3

SPECIAL_TOOL_OVERLAPS = {
    'wrench.png': 1,
    'hammer.png': 6,
    'drill.png': 19
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, INPUT_DIR_NAME)
MESSY_DIR = os.path.join(SCRIPT_DIR, MESSY_DIR_NAME)
NEAT_DIR = os.path.join(SCRIPT_DIR, NEAT_DIR_NAME)

os.makedirs(MESSY_DIR, exist_ok=True)
os.makedirs(NEAT_DIR, exist_ok=True)

def get_image_paths():
    image_paths = []
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        return []

    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.png'):
            image_paths.append(os.path.join(INPUT_DIR, filename))
    return image_paths


def check_overlap(box1, box2):
    x1, y1, x1_end, y1_end = box1
    x2, y2, x2_end, y2_end = box2
    return not (x1_end <= x2 or x2_end <= x1 or y1_end <= y2 or y2_end <= y1)


def generate_messy_image(objects_to_place, image_id):
    canvas = Image.new('RGBA', CANVAS_SIZE, (255, 255, 255, 255))
    placed_bboxes = []
    placed_objects = []

    random.shuffle(objects_to_place)

    for obj_path in objects_to_place:
        obj_image = Image.open(obj_path).convert("RGBA")
        rotation = random.randint(0, 359)
        rotated_obj = obj_image.rotate(rotation, expand=True)

        tight_bbox = rotated_obj.getbbox()
        if not tight_bbox: continue

        tight_width = tight_bbox[2] - tight_bbox[0]
        tight_height = tight_bbox[3] - tight_bbox[1]

        for attempt in range(MAX_PLACEMENT_ATTEMPTS):
            if CANVAS_SIZE[0] - tight_width < 0 or CANVAS_SIZE[1] - tight_height < 0:
                continue

            x = random.randint(0, CANVAS_SIZE[0] - tight_width)
            y = random.randint(0, CANVAS_SIZE[1] - tight_height)

            new_bbox = (x, y, x + tight_width, y + tight_height)

            is_overlapping = False
            for placed_bbox in placed_bboxes:
                if check_overlap(new_bbox, placed_bbox):
                    is_overlapping = True
                    break

            if not is_overlapping:
                paste_x = x - tight_bbox[0]
                paste_y = y - tight_bbox[1]

                canvas.paste(rotated_obj, (paste_x, paste_y), mask=rotated_obj)
                placed_bboxes.append(new_bbox)
                placed_objects.append(obj_path)
                break

    save_path = os.path.join(MESSY_DIR, f'messy_{image_id:06d}.png')
    canvas.save(save_path)
    return placed_objects

def create_tool_groups(objects_to_place):
    groups = collections.defaultdict(list)
    for obj_path in objects_to_place:
        groups[os.path.basename(obj_path)].append(Image.open(obj_path).convert("RGBA"))

    grouped_images = []
    filenames = list(groups.keys())
    random.shuffle(filenames)

    for filename in filenames:
        obj_list = groups[filename]
        if not obj_list: continue

        if 'ruler' in filename:
            for i in range(0, len(obj_list), 4):
                ruler_group = obj_list[i:i+4]
                if not ruler_group: continue
                max_width = max(img.width for img in ruler_group)
                total_height = sum(img.height for img in ruler_group) + (len(ruler_group) - 1) * RULER_SPACING
                group_img = Image.new('RGBA', (max_width, total_height), (0, 0, 0, 0))
                y_offset = 0
                for img in ruler_group:
                    group_img.paste(img, (0, y_offset), mask=img)
                    y_offset += img.height + RULER_SPACING
                grouped_images.append(group_img)

        elif filename in ['wrench.png', 'hammer.png']:
            overlap = SPECIAL_TOOL_OVERLAPS.get(filename, 0)
            combined_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            for i, img in enumerate(obj_list):
                rotated_img = img.rotate(180) if i % 2 != 0 else img
                new_w = combined_img.width + rotated_img.width - (overlap if i > 0 else 0)
                new_h = max(combined_img.height, rotated_img.height)
                temp = Image.new('RGBA', (new_w, new_h), (0, 0, 0, 0))
                temp.paste(combined_img, (0, new_h - combined_img.height), mask=combined_img)
                temp.paste(rotated_img, (combined_img.width - (overlap if i > 0 else 0), new_h - rotated_img.height), mask=rotated_img)
                combined_img = temp
            grouped_images.append(combined_img)

        elif filename == 'drill.png':
            for i in range(0, len(obj_list), 2):
                pair = obj_list[i:i+2]
                if not pair: continue
                img1 = pair[0]
                img2 = pair[1].rotate(180) if len(pair) > 1 else None
                overlap = SPECIAL_TOOL_OVERLAPS.get(filename, 0)
                w = img1.width + (img2.width - overlap if img2 else 0)
                h = max(img1.height, img2.height if img2 else 0)
                combined_img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
                combined_img.paste(img1, (0, h - img1.height), mask=img1)
                if img2: combined_img.paste(img2, (img1.width - overlap, h - img2.height), mask=img2)
                grouped_images.append(combined_img)

        else:
            combined_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            for i, img in enumerate(obj_list):
                new_w = combined_img.width + img.width + (HORIZONTAL_SPACING if i > 0 else 0)
                new_h = max(combined_img.height, img.height)
                temp = Image.new('RGBA', (new_w, new_h), (0, 0, 0, 0))
                temp.paste(combined_img, (0, new_h - combined_img.height), mask=combined_img)
                temp.paste(img, (combined_img.width + (HORIZONTAL_SPACING if i > 0 else 0), new_h - img.height), mask=img)
                combined_img = temp
            grouped_images.append(combined_img)

    return grouped_images


def layout_rows(canvas, items, start_x=0, start_y=0, max_w=CANVAS_SIZE[0]):
    current_x, current_y = start_x, start_y
    row_h = 0
    items.sort(key=lambda x: x.height, reverse=True)
    for img in items:
        if current_x + img.width > max_w:
            current_x = start_x
            current_y += row_h + HORIZONTAL_SPACING
            row_h = 0
        if current_y + img.height > CANVAS_SIZE[1]: continue
        canvas.paste(img, (current_x, current_y), mask=img)
        current_x += img.width + HORIZONTAL_SPACING
        row_h = max(row_h, img.height)

def layout_centered(canvas, items):
    temp_w, temp_h = 600, 600
    temp_canvas = Image.new('RGBA', (temp_w, temp_h), (0,0,0,0))
    total_area = sum(i.width * i.height for i in items)
    target_width = int(total_area ** 0.5 * 1.5)
    target_width = max(target_width, 150)
    current_x, current_y = 0, 0
    row_h = 0
    max_x, max_y = 0, 0
    items.sort(key=lambda x: x.height, reverse=True)
    for img in items:
        if current_x + img.width > target_width:
            current_x = 0
            current_y += row_h + HORIZONTAL_SPACING
            row_h = 0
        temp_canvas.paste(img, (current_x, current_y), mask=img)
        max_x = max(max_x, current_x + img.width)
        max_y = max(max_y, current_y + img.height)
        current_x += img.width + HORIZONTAL_SPACING
        row_h = max(row_h, img.height)
    content = temp_canvas.crop((0, 0, max_x, max_y))
    center_x = max(0, (CANVAS_SIZE[0] - content.width) // 2)
    center_y = max(0, (CANVAS_SIZE[1] - content.height) // 2)
    canvas.paste(content, (center_x, center_y), mask=content)

def layout_columns(canvas, items):
    rotated_items = [img.rotate(90, expand=True) for img in items]
    rotated_items.sort(key=lambda x: x.width, reverse=True)
    current_x, current_y = 0, 0
    col_w = 0
    for img in rotated_items:
        if current_y + img.height > CANVAS_SIZE[1]:
            current_y = 0
            current_x += col_w + HORIZONTAL_SPACING
            col_w = 0
        if current_x + img.width > CANVAS_SIZE[0]: continue
        canvas.paste(img, (current_x, current_y), mask=img)
        current_y += img.height + HORIZONTAL_SPACING
        col_w = max(col_w, img.width)

def layout_tight_clump(canvas, raw_object_paths):
    raw_images = [Image.open(p).convert("RGBA") for p in raw_object_paths]
    raw_images.sort(key=lambda x: x.height, reverse=True)
    temp_w, temp_h = 600, 600
    temp_canvas = Image.new('RGBA', (temp_w, temp_h), (0,0,0,0))
    total_area = sum(i.width * i.height for i in raw_images)
    target_width = int(total_area ** 0.5 * 1.2)
    target_width = max(target_width, 100)
    current_x, current_y = 0, 0
    row_h = 0
    max_x, max_y = 0, 0
    for img in raw_images:
        if current_x + img.width > target_width:
            current_x = 0
            current_y += row_h + HORIZONTAL_SPACING
            row_h = 0
        temp_canvas.paste(img, (current_x, current_y), mask=img)
        max_x = max(max_x, current_x + img.width)
        max_y = max(max_y, current_y + img.height)
        current_x += img.width + HORIZONTAL_SPACING
        row_h = max(row_h, img.height)
    content = temp_canvas.crop((0, 0, max_x, max_y))
    center_x = max(0, (CANVAS_SIZE[0] - content.width) // 2)
    center_y = max(0, (CANVAS_SIZE[1] - content.height) // 2)
    canvas.paste(content, (center_x, center_y), mask=content)

def layout_corners_by_type(canvas, raw_object_paths):
    type_map = collections.defaultdict(list)
    for p in raw_object_paths:
        type_map[os.path.basename(p)].append(p)
    tool_types = list(type_map.keys())
    random.shuffle(tool_types)
    anchors = [
        (0, 0, 1, 1),
        (CANVAS_SIZE[0], 0, -1, 1),
        (0, CANVAS_SIZE[1], 1, -1),
        (CANVAS_SIZE[0], CANVAS_SIZE[1], -1, -1),
        (CANVAS_SIZE[0]//2, CANVAS_SIZE[1]//2, 0, 0)
    ]
    for i, tool_type in enumerate(tool_types):
        if i >= len(anchors): break
        tool_paths = type_map[tool_type]
        neat_blocks = create_tool_groups(tool_paths)
        anchor_x, anchor_y, dir_x, dir_y = anchors[i]
        if dir_x == 0 and dir_y == 0:
            layout_centered(canvas, neat_blocks)
            continue
        current_offset_y = 0
        for img in neat_blocks:
            paste_x = anchor_x
            paste_y = anchor_y
            if dir_x == -1: paste_x -= img.width
            if dir_y == -1: paste_y -= img.height
            if dir_y == 1:
                paste_y += current_offset_y
            else:
                paste_y -= current_offset_y
            paste_x = max(0, min(paste_x, CANVAS_SIZE[0] - img.width))
            paste_y = max(0, min(paste_y, CANVAS_SIZE[1] - img.height))
            canvas.paste(img, (paste_x, paste_y), mask=img)
            current_offset_y += img.height + RULER_SPACING

def generate_neat_image(objects_to_place, image_id, variation_idx, strategy_name):
    canvas = Image.new('RGBA', CANVAS_SIZE, (255, 255, 255, 255))

    if strategy_name == 'tight_clump':
        layout_tight_clump(canvas, objects_to_place)
    elif strategy_name == 'corners':
        layout_corners_by_type(canvas, objects_to_place)
    else:
        grouped_items = create_tool_groups(objects_to_place)

        if strategy_name == 'rows':
            layout_rows(canvas, grouped_items)
        elif strategy_name == 'centered':
            layout_centered(canvas, grouped_items)
        elif strategy_name == 'columns':
            layout_columns(canvas, grouped_items)

    save_path = os.path.join(NEAT_DIR, f'neat_{image_id:06d}_{variation_idx}.png')
    canvas.save(save_path)

def main():
    image_paths = get_image_paths()

    if not image_paths:
        print("No images found in input directory. Exiting.")
        return

    print(f"Generating {NUM_OUTPUT_IMAGES} sets of images...")

    strategies = ['rows', 'columns', 'centered', 'tight_clump', 'corners']

    for i in range(NUM_OUTPUT_IMAGES):
        objects_to_place = []
        for obj_path in image_paths:
            num_copies = random.randint(0, MAX_COPIES_PER_OBJECT)
            for _ in range(num_copies):
                objects_to_place.append(obj_path)

        if not objects_to_place: continue

        messy_list = objects_to_place[:]
        generate_messy_image(messy_list, i)

        for v in range(VARIATIONS_PER_MESSY):
            neat_list = objects_to_place[:]
            random.shuffle(neat_list)

            strat = strategies[v % len(strategies)]
            generate_neat_image(neat_list, i, v, strategy_name=strat)

if __name__ == "__main__":
    main()