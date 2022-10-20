# It's chess in a day - one day to get the most functional 2p chess program up!
# Lets go!
import time

import numpy as np
from enum import Enum, auto
import cv2, dir, personal_utils, os, vlc, keyboard
import datetime


class OutputImg:
    total_image_size_yx = (720, 1280)
    label_offsets = .1
    color_value = 200

    grid_square_size_yx = int((total_image_size_yx[0] * (1 - label_offsets)) / 8), int((total_image_size_yx[0] * (1 - label_offsets)) / 8)
    x_label_offset = int(label_offsets * total_image_size_yx[0])
    icon_size_yx = (grid_square_size_yx[0] * 2 // 10), (grid_square_size_yx[0] * 2 // 10)
    icon_buffer = ((grid_square_size_yx[0] * 2) - icon_size_yx[0] * 8) // 10
    font_face = cv2.FONT_HERSHEY_COMPLEX_SMALL
    text_border_buffer = .4
    max_move_count = 128
    checkbox_checked_fill_percentage = .5

    ui_color = np.array([60, 60, 60], dtype=np.uint8)
    window_name = 'Chess'

    piece_abbreviations = {'knight': 'N',
                           'bishop': 'B',
                           'rook': 'R',
                           'queen': 'Q',
                           'king': 'K', }

    border_thickness = 2

    def __init__(self):
        self.menus = {}
        self.hover_color = np.array([0, 255, 255], dtype=np.uint8)
        self.board_img_bbox = np.array([0, self.grid_square_size_yx[0] * 8, self.x_label_offset, self.x_label_offset + self.grid_square_size_yx[1] * 8])
        self.img = np.zeros((self.total_image_size_yx[0], self.total_image_size_yx[1], 3), dtype=np.uint8)
        grid_square_templates = np.ones((2, 2, self.grid_square_size_yx[0], self.grid_square_size_yx[1], 3), dtype=np.uint8)
        self.square_color_white = np.array([255, 255, 244], dtype=np.uint8)
        hsv_color = np.array([[[int((269 / 359) * 179), 51, self.color_value]]], dtype=np.uint8)
        self.square_color_black = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR).flatten()
        self.valid_move_color = np.full(3, 255) - self.square_color_black
        self.mouse_callback_bboxs, self.mouse_callback_funcs = [], []
        grid_square_templates[0] = grid_square_templates[0] * self.square_color_white
        grid_square_templates[1] = grid_square_templates[1] * self.square_color_black
        for square in grid_square_templates:
            square[1] = square[0] * .35 + self.valid_move_color * .65
        self.grid_square_templates = grid_square_templates
        self.piece_imgs, self.piece_capture_imgs = self.return_img_arrays()
        self.piece_imgs_black_square_idxs, self.piece_imgs_drawn_move_idxs_w, self.piece_imgs_drawn_move_idxs_b = np.column_stack((np.argwhere(np.all(self.piece_imgs == self.square_color_black, axis=-1)))), \
                                                                                                                  np.column_stack((np.argwhere(np.all(self.piece_imgs == self.grid_square_templates[0, 1], axis=-1)))), \
                                                                                                                  np.column_stack((np.argwhere(np.all(self.piece_imgs == self.grid_square_templates[1, 1], axis=-1))))

        self.promotion_array_piece_idxs = np.array([[Pieces.queen.value, Pieces.rook.value, Pieces.bishop.value, Pieces.knight.value],
                                                    [Pieces.knight.value, Pieces.bishop.value, Pieces.rook.value, Pieces.queen.value]], dtype=np.uint8) + 1
        self.piece_idx_selected, self.drawn_moves = None, None
        self.drawn_previous_moves = None
        self.obstruction_next_vector_sign, self.obstruction_next_vector_i, self.obstruction_next_piece_yx = None, None, None
        self.promotion_ui_bbox_yx = None
        self.row_text, self.column_text = 'abcdefgh', '87654321'
        self.piece_location_str_grid = self.return_piece_location_text_grid()
        self.axes_color_idxs, self.axes_color_values = self.draw_grid_axes()
        self.move_count = 1

        # Defined in below function
        self.font_scale_thickness = None
        self.draw_notation_titles = True
        self.notation_tracker = [[], []]
        self.notation_tracker_display_max_count, self.menu_bbox, self.notation_y_idxs, self.notation_x_idxs, self.notation_scroll_bar_bbox, self.notation_cursor_x_idxs, self.notation_cursor_radius = self.return_notation_idxs(self.menus)
        self.notation_current_move_count, self.notation_current_turn_i = self.move_count, 0

        self.notation_scroll_bar_idx_offset_current, self.notation_scroll_bar_idx_offset_previous = 0, 0
        self.previous_board_states, self.previous_board_state_moves, self.previous_sound_states, self.current_board_state = np.zeros((self.max_move_count * 2 + 1, 3, 8, 8), dtype=np.uint8), \
                                                                                                                            np.zeros((self.max_move_count * 2 + 1, 2, 2), dtype=np.int16), np.zeros(self.max_move_count * 2 + 1, dtype=np.uint0), None
        self.previous_position_view_idx = None
        self.notation_cursor_current_xy = None
        self.draw_notation_selection_circle(1, 0, img=self.menus['notation']['img'])
        self.notation_img_template = self.menus['notation']['img'].copy()
        self.draw_notation_grid_onto_img(0)



        self.color_wheel, self.color_wheel_cursor_radius, self.current_color_cursor_location_yx = self.return_settings_menu(self.menu_bbox, self.menus)
        self.return_new_game_menu(self.menu_bbox, self.menus)

        self.capture_states = np.zeros((2, 15), dtype=np.uint8)
        self.current_menu = None
        self.board_color_idxs, self.img_color_idxs_black, self.board_color_idxs_moves_black, self.board_color_idxs_moves_black = None, None, None, None

        self.buttons, self.buttons_names, self.button_ranges = self.return_menu_buttons()
        self.sound_is_on = True
        self.current_menu, self.previous_drawn_board_state = 'notation', False
        self.fen_notation = None
        self.mouse_xy = None
        self.fisher = False
        self.status_bar_strs = ('White', 'Black')
        self.status_bar_current_text = None
        self.back_button_drawn = False
        self.previous_key_time, self.key_wait_time = time.perf_counter(), .025

    def return_piece_location_text_grid(self) -> np.ndarray:
        piece_locations = np.zeros((2, 8, 8), dtype=str)
        for row_i, row_text in enumerate(self.row_text):
            for column_i, column_text in enumerate(self.column_text):
                piece_locations[0, column_i, row_i] = row_text
                piece_locations[1, column_i, row_i] = column_text
        return piece_locations

    def return_notation_idxs(self, menu_dict):
        thickness, font_scale = 1, 1
        scroll_wheel_size = int(.025 * self.img.shape[0])
        scroll_wheel_buffer = int(.015 * self.img.shape[0])
        move_tracker_xy_coverage = .65, .55
        move_tracker_x, move_tracker_y = int((self.img.shape[1] - self.board_img_bbox[3]) * move_tracker_xy_coverage[0]), int(self.img.shape[0] * move_tracker_xy_coverage[1])
        move_tracker_x_offset = ((self.img.shape[1] - self.board_img_bbox[3]) - move_tracker_x) // 2

        move_tracker_bbox = np.array([self.board_img_bbox[0] + self.grid_square_size_yx[0], self.board_img_bbox[1] - self.grid_square_size_yx[0],
                                      self.board_img_bbox[3] + move_tracker_x_offset, self.board_img_bbox[3] + move_tracker_x_offset + move_tracker_x], dtype=np.uint16)
        scroll_wheel_bbox = np.array([move_tracker_bbox[0], move_tracker_bbox[1], move_tracker_bbox[3] + scroll_wheel_buffer, move_tracker_bbox[3] + scroll_wheel_buffer + scroll_wheel_size], dtype=np.uint16)
        # self.img[menu_bbox[0]:menu_bbox[1], menu_bbox[2]:menu_bbox[3]] = (255, 255, 255)
        # self.img[scroll_wheel_bbox[0]:scroll_wheel_bbox[1], scroll_wheel_bbox[2]:scroll_wheel_bbox[3]] = (255, 0, 0)

        max_turn_count_column_text = 'Move'
        max_notation = 'Qa1xe4++'
        max_total_text = max_turn_count_column_text + max_notation + max_notation
        text_buffer = .85
        text_pixels_x = int((((move_tracker_bbox[3] - move_tracker_bbox[2]) * text_buffer) - self.border_thickness * 2))
        text_pixels_x_text = int((((move_tracker_bbox[3] - move_tracker_bbox[2]) * text_buffer) - self.border_thickness * 2)) * .9

        text_width = 0
        font_scale = .001
        while text_pixels_x_text > text_width:
            font_scale += .001
            text_width = cv2.getTextSize(max_total_text, self.font_face, font_scale, thickness)[0][0]

        self.font_scale_thickness = font_scale, thickness

        text_size = cv2.getTextSize(max_total_text, self.font_face, font_scale, thickness)
        cursor_radius = text_size[0][1] + text_size[1]
        text_count = int((move_tracker_y // (text_size[0][1] + text_size[1])) * text_buffer)
        notation_text_size = cv2.getTextSize(max_notation, self.font_face, font_scale, thickness)
        turn_count_text_size = cv2.getTextSize(max_turn_count_column_text, self.font_face, font_scale, thickness)
        buffer_width = ((move_tracker_bbox[3] - move_tracker_bbox[2]) - (notation_text_size[0][0] * 2 + turn_count_text_size[0][0] + self.border_thickness * 2)) // 6
        buffer_width = (buffer_width - ((text_pixels_x - text_pixels_x_text)) // 6).astype(np.int32)
        buffer_pixels = np.cumsum(np.hstack((0, np.full(5, buffer_width))))

        bbox_x = np.array([[0, turn_count_text_size[0][0]],
                           [turn_count_text_size[0][0], turn_count_text_size[0][0] + notation_text_size[0][0]],
                           [turn_count_text_size[0][0] + notation_text_size[0][0], turn_count_text_size[0][0] + notation_text_size[0][0] * 2]])

        bbox_x[:, 0] += buffer_pixels[0::2]
        bbox_x[:, 1] += buffer_pixels[1::2]

        text_height = text_size[0][1] + text_size[1]

        text_height_cumsum = np.hstack((0, np.cumsum(np.full(self.max_move_count - 1, text_height))))
        grid_line_cumsum = np.cumsum(np.full(self.max_move_count - 1, self.border_thickness))
        buffer_height = ((move_tracker_bbox[1] - move_tracker_bbox[0]) - (text_height_cumsum[text_count] + grid_line_cumsum[text_count])) // text_count
        buffer_height_cumsum = np.cumsum(np.full(self.max_move_count, buffer_height))

        bbox_y = np.column_stack((text_height_cumsum[0:-1], text_height_cumsum[1:]))
        bbox_y[:, 1][0:] += buffer_height_cumsum[0:-1]
        bbox_y[:, 0][1:] += buffer_height_cumsum[1:-1]
        bbox_y[:, 1][1:] += grid_line_cumsum[0:-1]
        bbox_y[:, 0][1:] += grid_line_cumsum[1:]

        final_width, final_height = bbox_x[-1, 1], bbox_y[text_count, 1]
        bbox_x_offset, bbox_y_offset = (self.img.shape[1] - self.board_img_bbox[3] - final_width) // 2, (self.img.shape[0] - bbox_y[text_count, 1]) // 2

        move_tracker_bbox = np.array([self.board_img_bbox[0] + bbox_y_offset, self.board_img_bbox[0] + final_height + bbox_y_offset,
                                      self.board_img_bbox[3] + bbox_x_offset, self.board_img_bbox[3] + bbox_x_offset + final_width], dtype=np.int16)
        self.draw_border(self.img, move_tracker_bbox)
        notation_img = np.zeros((bbox_y[-1, 1] - bbox_y[0, 0], bbox_x[-1, 1], 3), dtype=np.uint8)

        for x_idx in range(bbox_x.shape[0] - 1):
            notation_img[:, bbox_x[x_idx, 1]:bbox_x[x_idx + 1, 0]] = (255, 255, 255)
        for y_idx in range(bbox_y.shape[0] - 1):
            bbox_mp_y = (bbox_y[y_idx, 1] + bbox_y[y_idx + 1, 0]) // 2
            start_y = bbox_mp_y - (self.border_thickness // 2)
            notation_img[start_y:start_y + self.border_thickness] = (255, 255, 255)

        for text, color, bbox in zip(('Move', 'White', 'Black'), (self.square_color_white, self.square_color_white, self.square_color_black), bbox_x):
            bbox_yx = np.hstack((bbox_y[0], bbox))
            self.draw_centered_text(notation_img, text, bbox_yx, color)

        # Draws current move circle on notation img

        move_tracker_bbox[0] += bbox_y[0, 1] - bbox_y[0, 1]
        menu_dict['notation'] = {'funcs': (self.return_to_previous_game_state,), 'bboxs': np.array([[move_tracker_bbox[0] + bbox_y[1, 0], move_tracker_bbox[1], move_tracker_bbox[2] + bbox_x[1, 0], move_tracker_bbox[2] + bbox_x[2, 1]]], dtype=np.uint16), 'img': notation_img}
        cx_cursors = int((bbox_x[1, 0] + cursor_radius // 2)), int((bbox_x[2, 0] + cursor_radius // 2))

        return text_count, move_tracker_bbox, bbox_y, bbox_x, scroll_wheel_bbox, cx_cursors, cursor_radius // 4

    def return_img_arrays(self):
        img_order = ('king', 'queen', 'bishop', 'knight', 'rook', 'pawn')
        img = cv2.imread(dir.chess_pieces_img, cv2.IMREAD_UNCHANGED)
        img_white = img[0:img.shape[0] // 2]
        img_black = img[img.shape[0] // 2:]
        buffer_percentage = 1 - .25
        draw_start_y_x = (self.grid_square_size_yx[0] - (self.grid_square_size_yx[0] * buffer_percentage)) // 2, (self.grid_square_size_yx[1] - (self.grid_square_size_yx[1] * buffer_percentage)) // 2
        img_resize_y_x = (int(self.grid_square_size_yx[1] * buffer_percentage), int(self.grid_square_size_yx[0] * buffer_percentage))

        # Piece_type, Piece_color, square_color, capture_bool (1 = capturable, colored,
        piece_imgs = np.zeros((len(img_order), 2, 2, 2, self.grid_square_size_yx[0], self.grid_square_size_yx[1], 3), dtype=np.uint8)
        capture_imgs = np.zeros((2, 6, self.icon_size_yx[0], self.icon_size_yx[1], 3), dtype=np.uint8)

        for piece_color_i, (img, border_color) in enumerate(zip((img_white, img_black), ((0, 0, 0), (255, 255, 255)))):
            img_mask = np.ascontiguousarray(img[0:, :, 3])
            contours, hier = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours = [contours[i] for i in range(0, len(contours))]

            contours.sort(key=len, reverse=True)
            contours = contours[0:len(img_order)]
            contour_cx_cys = np.zeros((len(contours), 2), dtype=np.uint16)

            for i, cnt in enumerate(contours):
                moments = cv2.moments(cnt)
                contour_cx_cys[i] = moments['m10'] / moments['m00'], moments['m01'] // moments['m00']

            contour_sort = list(np.argsort(contour_cx_cys[:, 0]))
            contours = [contours[i] for i in contour_sort]

            for contour_i, (piece_contour, piece_contour_name) in enumerate(zip(contours, img_order)):
                piece_bbox, piece_i, piece_icon_idx_count = cv2.boundingRect(piece_contour), Pieces[piece_contour_name].value, 0
                piece_native_size_img = img[piece_bbox[1]:piece_bbox[1] + piece_bbox[3], piece_bbox[0]:piece_bbox[0] + piece_bbox[2]]
                piece_native_size_mask = img_mask[piece_bbox[1]:piece_bbox[1] + piece_bbox[3], piece_bbox[0]:piece_bbox[0] + piece_bbox[2]]

                # personal_utils.image_show(piece_native_size_mask)
                piece_grid, piece_grid_mask = cv2.resize(piece_native_size_img, (img_resize_y_x[1], img_resize_y_x[0])), cv2.resize(piece_native_size_mask, (img_resize_y_x[1], img_resize_y_x[0]))
                piece_grid_mask_non_zero = np.column_stack(np.nonzero(piece_grid_mask))

                if piece_contour_name != 'king':
                    piece_capture_icon = cv2.resize(piece_grid_mask, (self.icon_size_yx[1], self.icon_size_yx[0]))
                    piece_capture_icon = cv2.cvtColor(piece_capture_icon, cv2.COLOR_GRAY2BGR)
                    if piece_color_i is black_i:
                        non_zero = np.nonzero(piece_capture_icon)
                        piece_capture_icon[non_zero[0], non_zero[1]] = self.square_color_black

                    if piece_contour_name == 'pawn':
                        capture_imgs[piece_color_i, 0] = piece_capture_icon
                    else:
                        capture_imgs[piece_color_i, Pieces[piece_contour_name].value] = piece_capture_icon

                for square_color_i in (white_i, black_i):
                    piece_mask_non_zero_relative = piece_grid_mask_non_zero - piece_grid_mask.shape

                    for capture_state in (non_capture_draw_i, capture_draw_i):
                        piece_template = self.grid_square_templates[square_color_i, capture_state].copy()
                        piece_template[piece_mask_non_zero_relative[:, 0] - int(draw_start_y_x[0]), piece_mask_non_zero_relative[:, 1] - int(draw_start_y_x[1])] = piece_grid[piece_grid_mask_non_zero[:, 0], piece_grid_mask_non_zero[:, 1]][:, 0:3]
                        piece_imgs[piece_i, piece_color_i, square_color_i, capture_state] = piece_template

                '''icon_nonzero = np.column_stack(np.nonzero(piece_capture_icon_mask))
                if icon_relative_idxs is None:
                    icon_relative_idxs = icon_nonzero - piece_capture_icon_mask.shape
                    icon_draw_points = piece_capture_icon_img[icon_nonzero]
                else:
                    icon_relative_idxs = np.vstack((icon_relative_idxs, (icon_nonzero - piece_capture_icon_mask.shape)))
                    icon_draw_points = np.vstack((icon_draw_points, piece_capture_icon_img[icon_nonzero]))
                icons_idx_ranges[piece_color_i, piece_icon_idx_count] = icon_idx_count, icon_idx_count + icon_nonzero.shape[0]
                icon_idx_count += icon_nonzero.shape[0]'''

        return piece_imgs, capture_imgs

    def return_draw_indicies_from_board_indicies(self, board_indicies_y_x):
        return np.column_stack((board_indicies_y_x[:, 0] * OutputImg.grid_square_size_yx[0],
                                (board_indicies_y_x[:, 0] + 1) * OutputImg.grid_square_size_yx[0],
                                board_indicies_y_x[:, 1] * OutputImg.grid_square_size_yx[1] + self.x_label_offset,
                                (board_indicies_y_x[:, 1] + 1) * OutputImg.grid_square_size_yx[1] + self.x_label_offset))

    def return_y_x_idx(self, mouse_x, mouse_y):
        return int((mouse_y / self.grid_square_size_yx[1])), \
               int((mouse_x - self.x_label_offset) / self.grid_square_size_yx[1])

    def return_menu_buttons(self) -> (np.ndarray, list, np.ndarray, (np.ndarray, np.ndarray)):
        # {Button:text, text_xy, bbox, func, img}
        buttons = {}

        def set_button_dict(button_name, bbox, func, text=None, text_xy=None, img=None, color_idxs=None):
            text_dict, img_dict = None, None
            if text is not None:
                text_dict = {'str': text, 'xy': text_xy}
            elif img is not None:
                img_dict = {'img': img, 'coloridxs': color_idxs}
            buttons[button_name] = {'text': text_dict, 'bbox': bbox, 'func': func, 'img': img_dict}

        def settings_cog():
            settings_cog_coverage = .05
            setting_cog_dim = int(settings_cog_coverage * self.img.shape[0])
            edge_buffer = int(.2 * setting_cog_dim)
            settings_cog_img = cv2.resize(cv2.imread(dir.settings_cog_img), (setting_cog_dim, setting_cog_dim))
            cog_non_zero = np.column_stack((np.nonzero(settings_cog_img)))
            settings_cog_img[cog_non_zero[:, 0], cog_non_zero[:, 1], 0:3] = self.square_color_black

            cog_back_button_bbox = np.array([0 + edge_buffer, 0 + edge_buffer + setting_cog_dim, self.img.shape[1] - edge_buffer - setting_cog_dim, self.img.shape[1] - edge_buffer], dtype=np.uint16)
            self.img[cog_back_button_bbox[0]:cog_back_button_bbox[1], cog_back_button_bbox[2]:cog_back_button_bbox[3]] = settings_cog_img
            back_button_img = np.ones_like(settings_cog_img) * self.square_color_black
            cv2.arrowedLine(back_button_img, (int(back_button_img.shape[1] * .75), back_button_img.shape[0] // 2), (int(back_button_img.shape[1] * .25), back_button_img.shape[0] // 2), (0, 0, 0), thickness=5, tipLength=.65,
                            line_type=cv2.LINE_AA)

            back_img_non_zero = np.column_stack((np.nonzero(back_button_img)))
            back_button_img[back_img_non_zero[0], back_img_non_zero[1]] = self.square_color_black
            set_button_dict('settings', cog_back_button_bbox, self.open_settings_menu, img=np.array([settings_cog_img, back_button_img]), color_idxs=(cog_non_zero[:, 0:2], back_img_non_zero[:, 0:2]))

        def new_game_button():
            text = 'New Game'
            text_size = cv2.getTextSize(text, self.font_face, self.font_scale_thickness[0], self.font_scale_thickness[1])
            mp_xy = (self.img.shape[1] * .5 + self.board_img_bbox[3] * 1.5) // 2 - (text_size[0][0] // 2), (self.img.shape[0] + self.board_img_bbox[1]) // 2 + (text_size[0][1] // 2),
            bbox = np.array([mp_xy[1] - text_size[0][1] - text_size[1] // 2, mp_xy[1] + text_size[1] // 2, mp_xy[0], mp_xy[0] + text_size[0][0]], dtype=np.uint16)
            self.img[bbox[0]:bbox[1], bbox[2]:bbox[3]] = self.square_color_black
            cv2.putText(self.img, text, (int(mp_xy[0]), int(mp_xy[1])), self.font_face, self.font_scale_thickness[0], (0, 0, 0), self.font_scale_thickness[1], lineType=cv2.LINE_AA)
            set_button_dict('newgame', bbox, self.open_new_game_menu, text=text, text_xy=(int(mp_xy[0]), int(mp_xy[1])))

        def export_moves_to_json():
            text = 'Export Notation'
            text_size = cv2.getTextSize(text, self.font_face, self.font_scale_thickness[0], self.font_scale_thickness[1])
            mp_xy = (self.img.shape[1] * 1.5 + self.board_img_bbox[3] * .5) // 2 - (text_size[0][0] // 2), (self.img.shape[0] + self.board_img_bbox[1]) // 2 + (text_size[0][1] // 2),
            bbox = np.array([mp_xy[1] - text_size[0][1] - text_size[1] // 2, mp_xy[1] + text_size[1] // 2, mp_xy[0], mp_xy[0] + text_size[0][0]], dtype=np.uint16)
            self.img[bbox[0]:bbox[1], bbox[2]:bbox[3]] = self.square_color_black
            cv2.putText(self.img, text, (int(mp_xy[0]), int(mp_xy[1])), self.font_face, self.font_scale_thickness[0], (0, 0, 0), self.font_scale_thickness[1], lineType=cv2.LINE_AA)
            set_button_dict('notation', bbox, self.export_moves, text=text, text_xy=(int(mp_xy[0]), int(mp_xy[1])))

        settings_cog(), new_game_button(), export_moves_to_json()
        button_ranges, button_names = [], []
        for key in buttons:
            button_names.append(key)
            button_ranges.append(buttons[key]['bbox'])
            self.draw_border(self.img, buttons[key]['bbox'])

        return buttons, button_names, np.vstack(button_ranges)

    def return_settings_menu(self, bbox, menu_dict):
        settings_template_img = np.zeros((bbox[1] - bbox[0], bbox[3] - bbox[2], 3), dtype=np.uint8)
        buffer = .85
        sat_min, sat_range = 35, 175
        hsv_wheel_size = int(settings_template_img.shape[1] * buffer)
        x_offset, y_offset = int((1 - buffer) * settings_template_img.shape[1]) // 2, (settings_template_img.shape[0] - hsv_wheel_size) // 8
        color_wheel_img = np.zeros((hsv_wheel_size, hsv_wheel_size, 3), dtype=np.uint8)
        cv2.circle(color_wheel_img, (hsv_wheel_size // 2, hsv_wheel_size // 2), (hsv_wheel_size // 2) - 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        hsv_yx = np.column_stack(np.nonzero(cv2.cvtColor(color_wheel_img, cv2.COLOR_BGR2GRAY)))
        hsv_y_x_idxs = hsv_yx - (hsv_wheel_size // 2, hsv_wheel_size // 2)
        hsv_values = np.arctan2(hsv_y_x_idxs[:, 0], hsv_y_x_idxs[:, 1]) * 180 / np.pi
        hsv_values[np.argwhere(hsv_values < 0)] = 180 + np.abs(hsv_values[np.argwhere(hsv_values < 0)] + 180)
        hsv_values *= (179 / 359)
        dist = np.linalg.norm(hsv_y_x_idxs - (0, 0), axis=1)
        dist_saturation_values = (dist / dist.max() * sat_range) + sat_min
        value_values = np.full_like(dist_saturation_values, self.color_value)
        hsv_wheel_values = np.column_stack((hsv_values, dist_saturation_values, value_values)).astype(np.uint8)
        bgr_values = cv2.cvtColor(np.array([hsv_wheel_values]), cv2.COLOR_HSV2BGR)[0]
        hsv_starting_color = cv2.cvtColor(np.array([[self.square_color_black]]), cv2.COLOR_BGR2HSV).flatten()

        starting_radius = int((hsv_starting_color[1] / (sat_min + sat_range)) * hsv_wheel_size)
        starting_cursor_xy = int(starting_radius * np.cos((hsv_starting_color[0] / 179) * 360)) + hsv_wheel_size // 2, int(starting_radius * np.sin((hsv_starting_color[0] / 179) * 360)) + hsv_wheel_size // 2

        starting_cursor_idx = np.argmin(np.sum(np.abs(hsv_wheel_values - self.square_color_black), axis=1))
        color_wheel_img[hsv_yx[:, 0], hsv_yx[:, 1]] = bgr_values

        current_menu_wheel_img, current_cursor_mp = color_wheel_img.copy(), hsv_yx[starting_cursor_idx]

        hsv_cursor_selector_radius = max(int(hsv_wheel_size * .0125), 1)
        cv2.circle(current_menu_wheel_img, starting_cursor_xy, hsv_cursor_selector_radius, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        sound_bbox_length = max(int(hsv_wheel_size * .2), 1)

        self.checkbox_size_all_fill = int(sound_bbox_length), int((sound_bbox_length * self.checkbox_checked_fill_percentage) // 4)
        bbox_checkbox = self.draw_initial_options_checkbox_and_return_bbox('Sound', settings_template_img, vertical_placement_percentage=.25, is_checked=True)
        color_wheel_bbox = np.array([settings_template_img.shape[0] - y_offset - current_menu_wheel_img.shape[0], settings_template_img.shape[0] - y_offset, x_offset, x_offset + current_menu_wheel_img.shape[1]])
        settings_template_img[color_wheel_bbox[0]:color_wheel_bbox[1], color_wheel_bbox[2]:color_wheel_bbox[3]] = current_menu_wheel_img

        bboxs = np.vstack((np.array([self.menu_bbox[0] + color_wheel_bbox[0], self.menu_bbox[0] + color_wheel_bbox[1],
                                     self.menu_bbox[2] + color_wheel_bbox[2], self.menu_bbox[2] + color_wheel_bbox[3]]),
                           np.array([self.menu_bbox[0] + bbox_checkbox[0], self.menu_bbox[0] + bbox_checkbox[1],
                                     self.menu_bbox[2] + bbox_checkbox[2], self.menu_bbox[2] + bbox_checkbox[3]])))

        menu_dict['settings'] = {'funcs': [self.set_accent_color, self.set_sound_setting], 'bboxs': bboxs, 'img': settings_template_img}

        return color_wheel_img, hsv_cursor_selector_radius, (starting_cursor_xy[1], starting_cursor_xy[0])

    def return_new_game_menu(self, bbox, menu_dict):
        menu_img = np.zeros((bbox[1] - bbox[0], bbox[3] - bbox[2], 3), dtype=np.uint8)
        chess960 = 'Chess 960'
        bbox_checkbox = self.draw_initial_options_checkbox_and_return_bbox(chess960, menu_img, vertical_placement_percentage=.25, is_checked=False)
        bbox_start_game = np.array([menu_img.shape[0] // 2, menu_img.shape[0], menu_img.shape[1] // 2, menu_img.shape[1] // 2], dtype=np.uint16)
        bbox_start = self.draw_centered_text(menu_img, 'Start Game', bbox_start_game, (0, 0, 0), fill_color=self.square_color_black, return_bbox=True)
        self.draw_border(menu_img, bbox_start)
        menu_dict['newgame'] = {'funcs': (self.set_fisher, self.start_new_game), 'img': menu_img, 'bboxs': np.vstack((bbox_checkbox + np.repeat((self.menu_bbox[0], self.menu_bbox[2]), 2),
                                                                                                                      bbox_start + np.repeat((self.menu_bbox[0], self.menu_bbox[2]), 2)))}

    def open_new_game_menu(self):
        if self.current_menu == 'settings':
            self.finalize_accent_color()
        elif self.move_count != self.notation_current_move_count:
            self.set_current_game_state()

        self.current_menu = 'newgame'
        button_bbox = self.buttons['settings']['bbox']
        self.img[button_bbox[0]:button_bbox[1], button_bbox[2]:button_bbox[3]] = self.buttons['settings']['img']['img'][1]
        self.draw_menu(self.menus['newgame']['img'])

    def open_settings_menu(self):
        global turn_i_current_opponent
        if self.current_menu == 'settings':
            self.finalize_accent_color()
            self.current_menu = 'notation'
            self.draw_notation_grid_onto_img(self.move_count)
            self.draw_back_button(undraw=True)

        elif self.current_menu == 'notation':
            if self.previous_position_view_idx is None:
                self.draw_menu(self.menus['settings']['img'])
                self.board_color_idxs = np.argwhere(np.all(self.img[self.board_img_bbox[0]:self.board_img_bbox[1], self.board_img_bbox[2]:self.board_img_bbox[3] + self.icon_buffer + self.icon_size_yx[1]] == self.square_color_black, axis=-1)) + (self.board_img_bbox[0], self.board_img_bbox[2])
                self.board_color_idxs_moves_black = np.argwhere(np.all(self.img[self.board_img_bbox[0]:self.board_img_bbox[1], self.board_img_bbox[2]:self.board_img_bbox[3] + self.icon_buffer + self.icon_size_yx[1]] == self.grid_square_templates[0, 1, 0, 0], axis=-1)) + (
                    self.board_img_bbox[0], self.board_img_bbox[2])
                self.board_color_idxs_moves_white = np.argwhere(np.all(self.img[self.board_img_bbox[0]:self.board_img_bbox[1], self.board_img_bbox[2]:self.board_img_bbox[3] + self.icon_buffer + self.icon_size_yx[1]] == self.grid_square_templates[1, 1, 0, 0], axis=-1)) + (
                    self.board_img_bbox[0], self.board_img_bbox[2])
                self.current_menu = 'settings'
                self.draw_back_button()
            else:
                self.previous_position_view_idx = None
                self.set_current_game_state()

        elif self.current_menu == 'newgame':
            self.current_menu = 'notation'
            self.draw_back_button(undraw=True)
            if turn_i_current_opponent[0] == 1:
                self.draw_notation_grid_onto_img(self.move_count)
            else:
                self.draw_notation_grid_onto_img(self.move_count - 1)

    def draw_back_button(self, undraw=False):
        button_bbox = self.buttons['settings']['bbox']
        if not undraw:
            self.img[button_bbox[0]:button_bbox[1], button_bbox[2]:button_bbox[3]] = self.buttons['settings']['img']['img'][1]
            self.back_button_drawn = True
        else:
            self.img[button_bbox[0]:button_bbox[1], button_bbox[2]:button_bbox[3]] = self.buttons['settings']['img']['img'][0]
            self.back_button_drawn = False

    def start_new_game(self):
        global new_game
        new_game = True
        self.move_count = 0
        self.open_settings_menu()
        self.current_menu = 'notation'
        self.menus['notation']['img'] = self.notation_img_template.copy()
        self.draw_notation_grid_onto_img(self.move_count)

    def set_sound_setting(self):
        sound_bbox = self.menus['settings']['bboxs'][1]
        relative_bbox = sound_bbox - (self.menu_bbox[0], self.menu_bbox[0], self.menu_bbox[2], self.menu_bbox[2])
        self.fill_checkbox(self.menus['settings']['img'], relative_bbox, self.sound_is_on)
        self.fill_checkbox(self.img, sound_bbox, self.sound_is_on)
        self.sound_is_on = not self.sound_is_on

    def set_fisher(self):
        bbox_960 = self.menus['newgame']['bboxs'][0]
        relative_bbox = bbox_960 - (self.menu_bbox[0], self.menu_bbox[0], self.menu_bbox[2], self.menu_bbox[2])

        self.fill_checkbox(self.menus['newgame']['img'], relative_bbox, self.fisher)
        self.fill_checkbox(self.img, bbox_960, self.fisher)
        self.fisher = not self.fisher

    def set_fen_notation_str(self, row_values=None):
        if self.fisher:
            fen_str_w = ''
            for piece_id in row_values:
                fen_str_w = f'{fen_str_w}{self.piece_abbreviations[Pieces(piece_id - 1).name]}'
            self.fen_notation = f'{fen_str_w.lower()}/pppppppp/8/8/8/8/PPPPPPPP/{fen_str_w} w KQkq - 0 1'
        else:
            self.fen_notation = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

    def set_accent_color(self):
        bbox_draw = self.menus['settings']['bboxs'][0]
        yx_adj = self.mouse_xy[1] - bbox_draw[0], self.mouse_xy[0] - bbox_draw[2]
        if yx_adj[0] < self.color_wheel.shape[0] and yx_adj[1] < self.color_wheel.shape[1]:
            color_selected = self.color_wheel[yx_adj[0], yx_adj[1]]
            if np.all(color_selected != 0):
                self.square_color_black = color_selected
                self.valid_move_color = np.full(3, 255) - self.square_color_black
                self.grid_square_templates[1] = self.square_color_black
                color_wheel_bbox = self.menus['settings']['bboxs'][0]
                relative_bbox = color_wheel_bbox - (self.menu_bbox[0], self.menu_bbox[0], self.menu_bbox[2], self.menu_bbox[2])

                for square in self.grid_square_templates:
                    square[1] = square[0] * .35 + self.valid_move_color * .65

                color_array_cursor_mask = np.zeros((self.color_wheel.shape[0], self.color_wheel.shape[1]), dtype=np.uint8)
                cv2.circle(color_array_cursor_mask, (self.current_color_cursor_location_yx[1], self.current_color_cursor_location_yx[0]), self.color_wheel_cursor_radius, 255, -1, lineType=cv2.LINE_AA)
                cv2.circle(color_array_cursor_mask, (self.current_color_cursor_location_yx[1], self.current_color_cursor_location_yx[0]), self.color_wheel_cursor_radius, 255, 1, lineType=cv2.LINE_AA)
                color_wheel_non_zero = np.column_stack(np.nonzero(color_array_cursor_mask))
                color_wheel_values = self.color_wheel[color_wheel_non_zero[:, 0], color_wheel_non_zero[:, 1]]

                # Clears cursor off of the current setting img and the output img
                for img, bbox in zip((self.menus['settings']['img'], self.img), (relative_bbox, bbox_draw)):
                    color_wheel_idxs = color_wheel_non_zero + (bbox[0], bbox[2])
                    img[color_wheel_idxs[:, 0], color_wheel_idxs[:, 1]] = color_wheel_values
                    new_cursor_xy = int(bbox[2] + yx_adj[1]), int(bbox[0] + yx_adj[0])
                    cv2.circle(img, new_cursor_xy, self.color_wheel_cursor_radius, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                cv2.imshow(self.window_name, self.img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()

                self.current_color_cursor_location_yx = yx_adj
                for draw_idxs in (self.board_color_idxs, self.axes_color_idxs):
                    if draw_idxs.shape[0] != 0:
                        self.img[draw_idxs[:, 0], draw_idxs[:, 1]] = color_selected
                    if draw_idxs is self.axes_color_idxs:
                        # Keeps text sharp
                        self.img[draw_idxs[:, 0], draw_idxs[:, 1]] = color_selected * self.axes_color_values

                for draw_idxs, color in zip((self.board_color_idxs_moves_white, self.board_color_idxs_moves_black), (self.grid_square_templates[0, 1, 0, 0], self.grid_square_templates[1, 1, 0, 0])):
                    self.img[draw_idxs[:, 0], draw_idxs[:, 1]] = color

                for key in self.buttons:
                    if self.buttons[key]['text']:
                        bbox = self.buttons[key]['bbox']
                        self.img[bbox[0]:bbox[1], bbox[2]:bbox[3]] = color_selected
                        cv2.putText(self.img, self.buttons[key]['text']['str'], self.buttons[key]['text']['xy'], self.font_face, self.font_scale_thickness[0], (0, 0, 0), self.font_scale_thickness[1], cv2.LINE_AA)
                    else:
                        for img, color_idxs in zip(self.buttons['settings']['img']['img'], self.buttons['settings']['img']['coloridxs']):
                            img[color_idxs[:, 0], color_idxs[:, 1]] = color_selected

                # Draws Back Button
                button_bbox = self.buttons['settings']['bbox']
                self.img[button_bbox[0]:button_bbox[1], button_bbox[2]:button_bbox[3]] = self.buttons['settings']['img']['img'][1]

                if self.sound_is_on:
                    sound_bbox = self.menus['settings']['bboxs'][1]
                    relative_bbox = sound_bbox - (self.menu_bbox[0], self.menu_bbox[0], self.menu_bbox[2], self.menu_bbox[2])
                    self.fill_checkbox(self.menus['settings']['img'], relative_bbox, uncheck=False)
                    self.fill_checkbox(self.img, sound_bbox, uncheck=False)

            if turn_i_current_opponent[0] == black_i:
                if self.status_bar_current_text is not None:
                    self.draw_board_status(self.status_bar_current_text, text_color=color_selected)

    def finalize_accent_color(self):
        for piece_idxs, color in zip((self.piece_imgs_black_square_idxs, self.piece_imgs_drawn_move_idxs_w, self.piece_imgs_drawn_move_idxs_b), (self.square_color_black, self.grid_square_templates[0, 1, 0, 0], self.grid_square_templates[1, 1, 0, 0])):
            self.piece_imgs[piece_idxs[0], piece_idxs[1], piece_idxs[2], piece_idxs[3], piece_idxs[4], piece_idxs[5]] = color
        self.current_menu = 'notation'
        button_bbox = self.buttons['settings']['bbox']
        self.img[button_bbox[0]:button_bbox[1], button_bbox[2]:button_bbox[3]] = self.buttons['settings']['img']['img'][0]
        x_idxs = self.notation_x_idxs[2]
        for y_idx, text in zip(self.notation_y_idxs[1:1 + len(self.notation_tracker[1])], self.notation_tracker[1]):
            # Draws the selected accent color onto the template
            self.draw_centered_text(self.menus['notation']['img'], text, np.hstack((y_idx, x_idxs)), text_color=self.square_color_black, fill_color=(0, 0, 0))

        relative_bbox = self.menus['newgame']['bboxs'][1] - (self.menu_bbox[0] - 1, self.menu_bbox[0] - 1, self.menu_bbox[2] + 1, self.menu_bbox[2])
        self.draw_centered_text(self.menus['newgame']['img'], 'Start Game', relative_bbox, (0, 0, 0), fill_color=self.square_color_black)

        if self.fisher:
            chess_960_bbox = self.menus['newgame']['bboxs'][0]
            relative_bbox = chess_960_bbox - (self.menu_bbox[0], self.menu_bbox[0], self.menu_bbox[2], self.menu_bbox[2])
            self.fill_checkbox(self.menus['newgame']['img'], relative_bbox, uncheck=False)

        self.draw_notation_titles = True

    def draw_board(self, position_array_draw_idxs_y_x, position_array, pre_move=False):
        draw_idxs = self.return_draw_indicies_from_board_indicies(position_array_draw_idxs_y_x)
        pieces = np.argwhere(position_array[0, position_array_draw_idxs_y_x[:, 0], position_array_draw_idxs_y_x[:, 1]] != 0).flatten()
        empty_spaces = np.argwhere(position_array[0, position_array_draw_idxs_y_x[:, 0], position_array_draw_idxs_y_x[:, 1]] == 0).flatten()
        if pre_move:
            final_color_i = 1
        else:
            final_color_i = 0
        if pieces.shape[0] != 0:
            piece_draw_is = np.vstack((position_array[0:3, position_array_draw_idxs_y_x[:, 0][pieces], position_array_draw_idxs_y_x[:, 1][pieces]], np.full(pieces.shape[0], final_color_i)))
            for draw_i, img_i in zip(pieces, piece_draw_is.T):
                self.img[draw_idxs[draw_i, 0]: draw_idxs[draw_i, 1], draw_idxs[draw_i, 2]: draw_idxs[draw_i, 3]] = self.piece_imgs[img_i[0] - 1, img_i[1], img_i[2], img_i[3]]

        if empty_spaces.shape[0] != 0:
            space_draw_is = position_array[2, position_array_draw_idxs_y_x[:, 0][empty_spaces], position_array_draw_idxs_y_x[:, 1][empty_spaces]]
            for draw_i, img_i in zip(empty_spaces, space_draw_is):
                self.img[draw_idxs[draw_i, 0]: draw_idxs[draw_i, 1], draw_idxs[draw_i, 2]: draw_idxs[draw_i, 3]] = self.grid_square_templates[img_i, final_color_i]
            pass

    def draw_centered_text(self, img, text, bbox, text_color, fill_color=None, return_bbox=False) -> np.ndarray or None:
        img[bbox[0]:bbox[1], bbox[2]:bbox[3]] = 0, 0, 0

        text_size = cv2.getTextSize(text, self.font_face, self.font_scale_thickness[0], self.font_scale_thickness[1])
        mp_xy = (bbox[3] + bbox[2] - text_size[0][0]) // 2, ((bbox[0] + bbox[1]) + text_size[0][1] - text_size[1] // 2) // 2
        bbox = np.array([mp_xy[1] - text_size[0][1] - text_size[1] // 2, mp_xy[1] + text_size[1] // 2, mp_xy[0], mp_xy[0] + text_size[0][0]], dtype=np.uint16)
        if fill_color is not None:
            img[bbox[0]:bbox[1], bbox[2]:bbox[3]] = fill_color

        cv2.putText(img, text, mp_xy, self.font_face, self.font_scale_thickness[0], (int(text_color[0]), int(text_color[1]), int(text_color[2])), self.font_scale_thickness[1], lineType=cv2.LINE_AA)

        if return_bbox:
            return bbox

    def draw_grid_axes(self):
        font_scale, thickness, = 1.0, 2
        start_points_x, start_points_y = np.linspace(self.board_img_bbox[2], self.board_img_bbox[3], num=9, endpoint=True, dtype=np.uint16), \
                                         np.linspace(self.board_img_bbox[0], self.board_img_bbox[1], num=9, endpoint=True, dtype=np.uint16)
        char_max_size_idx = np.argmax(np.array([cv2.getTextSize(char, self.font_face, font_scale, thickness)[0][0] for char in self.column_text]))
        max_size_char = self.column_text[char_max_size_idx]
        char_grid_max_size = int(self.grid_square_size_yx[1] * self.text_border_buffer)
        char_grid_x_y_offset = (self.grid_square_size_yx[1] - char_grid_max_size) // 2
        char_grid_size = 0

        while char_grid_max_size > char_grid_size:
            font_scale += .1
            char_grid_size = cv2.getTextSize(max_size_char, self.font_face, font_scale, thickness)[0][0]

        for i, x_start in enumerate(start_points_x[0:-1]):
            text_size = cv2.getTextSize(self.row_text[i], self.font_face, font_scale, thickness)
            text_xy = x_start + char_grid_x_y_offset, start_points_y[-1] + char_grid_x_y_offset + text_size[0][1] - text_size[1] // 2
            cv2.putText(self.img, self.row_text[i], text_xy, self.font_face, font_scale, (int(self.square_color_black[0]), int(self.square_color_black[1]), int(self.square_color_black[2])), thickness, lineType=cv2.LINE_AA)

        for i, y_start in enumerate(start_points_y[0:-1]):
            text_size = cv2.getTextSize(self.column_text[i], self.font_face, font_scale, thickness)
            text_xy = char_grid_x_y_offset, y_start + char_grid_x_y_offset + text_size[0][1] // 2 + text_size[1]
            cv2.putText(self.img, self.column_text[i], text_xy, self.font_face, font_scale, (int(self.square_color_black[0]), int(self.square_color_black[1]), int(self.square_color_black[2])), thickness, lineType=cv2.LINE_AA)

        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        text_color_idxs = np.column_stack(np.nonzero(img_gray))
        text_color_value_idxs = self.img[text_color_idxs[:, 0], text_color_idxs[:, 1]] / self.square_color_black
        return text_color_idxs, text_color_value_idxs

    def draw_promotion_selector(self, idx_x, board_state, current_turn_i):
        if current_turn_i == black_i:
            promotion_y_idx_start_stop = 4, 8
        else:
            promotion_y_idx_start_stop = 0, 4

        promotion_ui_bbox_yx = np.array([promotion_y_idx_start_stop[0] * self.grid_square_size_yx[0], promotion_y_idx_start_stop[1] * (self.grid_square_size_yx[0] + 1),
                                         (idx_x * self.grid_square_size_yx[1]) + self.x_label_offset, ((idx_x + 1) * self.grid_square_size_yx[1]) + self.x_label_offset], dtype=np.uint16)
        self.promotion_ui_bbox_yx = promotion_ui_bbox_yx.copy()

        promotion_pos_array = board_state.copy()
        promotion_pos_array[0, promotion_y_idx_start_stop[0]:promotion_y_idx_start_stop[1], idx_x] = self.promotion_array_piece_idxs[current_turn_i]
        promotion_pos_array[1, promotion_y_idx_start_stop[0]:promotion_y_idx_start_stop[1], idx_x] = current_turn_i
        draw_idxs = np.column_stack((np.arange(promotion_y_idx_start_stop[0], promotion_y_idx_start_stop[1]),
                                     np.full_like(self.promotion_array_piece_idxs[current_turn_i], idx_x)))
        self.draw_board(draw_idxs, promotion_pos_array, pre_move=True)
        self.drawn_moves = draw_idxs

    def draw_notation_img(self, turn_i, yx_initial, yx_selected, piece, row_identifier=False, column_identifier=False, piece_captured=False, piece_promotion=None, castle_text=None, game_state_text=None):
        if castle_text is None:
            if piece != Pieces.pawn:
                text = f'{self.piece_abbreviations[piece.name]}'
                if column_identifier:
                    text = f'{text}{self.piece_location_str_grid[0, yx_initial[0], yx_initial[1]]}'
                if row_identifier:
                    text = f'{text}{self.piece_location_str_grid[1, yx_initial[0], yx_initial[1]]}'
                if piece_captured:
                    text = f'{text}x{self.piece_location_str_grid[0, yx_selected[0], yx_selected[1]]}{self.piece_location_str_grid[1, yx_selected[0], yx_selected[1]]}'
                else:
                    text = f'{text}{self.piece_location_str_grid[0, yx_selected[0], yx_selected[1]]}{self.piece_location_str_grid[1, yx_selected[0], yx_selected[1]]}'
            else:
                text = f'{self.piece_location_str_grid[0, yx_selected[0], yx_selected[1]] + self.piece_location_str_grid[1, yx_selected[0], yx_selected[1]]}'
                if piece_captured:
                    text = f'{self.piece_location_str_grid[0, yx_initial[0], yx_initial[1]] + self.piece_location_str_grid[1, yx_initial[0], yx_initial[1]]}x{text}'
                if piece_promotion:
                    text = f'{text}{self.piece_abbreviations[piece_promotion.name]}'
        else:
            text = castle_text

        if game_state_text:
            text = f'{text}{game_state_text}'

        row_idx, column_idx = (self.move_count // 2) + 1, (self.move_count % 2) + 1
        if column_idx == 1:
            color = self.square_color_white
            self.draw_centered_text(self.menus['notation']['img'], str(row_idx), np.hstack((self.notation_y_idxs[row_idx], self.notation_x_idxs[column_idx - 1])), color)
            self.draw_notation_selection_circle(row_idx, column_idx)
        else:
            color = self.square_color_black
            self.draw_notation_selection_circle(row_idx + 1, 0)

        self.draw_centered_text(self.menus['notation']['img'], text, np.hstack((self.notation_y_idxs[row_idx], self.notation_x_idxs[column_idx])), color)
        self.draw_notation_grid_onto_img(self.move_count)
        self.notation_tracker[turn_i].append(text)
        self.move_count += 1
        self.notation_current_move_count = self.move_count

    def draw_notation_selection_circle(self, y_idx, x_idx, img =None):
        if img is None:
            img = self.menus['notation']['img']

        if self.notation_cursor_current_xy is not None:
            cv2.circle(img, self.notation_cursor_current_xy, self.notation_cursor_radius + 1, (0, 0, 0), -1, lineType=cv2.LINE_AA)

        self.notation_cursor_current_xy = self.notation_cursor_x_idxs[x_idx], int(np.mean(self.notation_y_idxs[y_idx]))
        cv2.circle(img, self.notation_cursor_current_xy, self.notation_cursor_radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    def draw_notation_grid_onto_img(self, move_count=None):
        if move_count is None:
            move_count = self.move_count

        notation_row_idx = 1 + (move_count // 2)
        notation_column_idx = move_count % 2

        if self.draw_notation_titles:
            for bbox_x, text, text_color in zip(self.notation_x_idxs, ('Move', 'White', 'Black'),
                                                ((255, 255, 255), (255, 255, 255), (int(self.square_color_black[0]), int(self.square_color_black[1]), int(self.square_color_black[2])))):
                self.draw_centered_text(self.menus['notation']['img'], text, np.hstack((self.notation_y_idxs[0], bbox_x)), text_color, fill_color=(0, 0, 0))
                if text == 'Black':
                    self.draw_centered_text(self.notation_img_template, text, np.hstack((self.notation_y_idxs[0], bbox_x)), text_color, fill_color=(0, 0, 0))
            self.draw_notation_titles = False

        self.img[self.menu_bbox[0]:self.menu_bbox[0] + self.notation_y_idxs[1, 0], self.menu_bbox[2]:self.menu_bbox[3]] = self.menus['notation']['img'][self.notation_y_idxs[0, 0]:self.notation_y_idxs[1, 0]]

        if notation_row_idx < self.notation_tracker_display_max_count:
            start_i, end_i = 0, notation_row_idx
            img_range_y = self.notation_y_idxs[start_i + self.notation_tracker_display_max_count, 1] - self.notation_y_idxs[start_i, 0]
            self.img[self.menu_bbox[0]:self.menu_bbox[0] + img_range_y, self.menu_bbox[2]:self.menu_bbox[3]] = self.menus['notation']['img'][self.notation_y_idxs[start_i, 0]: self.notation_y_idxs[start_i + self.notation_tracker_display_max_count, 1]]
        else:
            if notation_column_idx == 0:
                start_i = notation_row_idx - (self.notation_tracker_display_max_count - 1)
                end_i = (start_i + self.notation_tracker_display_max_count) - 1
                self.img[self.menu_bbox[0] + self.notation_y_idxs[1, 0]:self.menu_bbox[1], self.menu_bbox[2]:self.menu_bbox[3]] = self.menus['notation']['img'][self.notation_y_idxs[start_i, 0]: self.notation_y_idxs[end_i, 1]]
            else:
                start_i = notation_row_idx - (self.notation_tracker_display_max_count - 2)
                end_i = (start_i + self.notation_tracker_display_max_count) - 1
                self.img[self.menu_bbox[0] + self.notation_y_idxs[1, 0]:self.menu_bbox[1], self.menu_bbox[2]:self.menu_bbox[3]] = self.menus['notation']['img'][self.notation_y_idxs[start_i, 0]: self.notation_y_idxs[end_i, 1]]

    def notation_keyboard_handler(self, key: keyboard.KeyboardEvent):
        #Slight delay to prevent double inputs
        if self.current_menu == 'notation' and time.perf_counter() - self.previous_key_time >= self.key_wait_time:
            key_pressed, previous_state_idx = key.name, self.notation_current_move_count
            if key_pressed == 'up':
                previous_state_idx -= 2
            elif key_pressed == 'down':
                previous_state_idx += 2
            elif key_pressed == 'left':
                previous_state_idx -= 1
            elif key_pressed == 'right':
                previous_state_idx += 1

            if 0 <= previous_state_idx <= self.move_count:
                self.notation_current_move_count = previous_state_idx
                self.set_previous_game_state(previous_state_idx)
                self.draw_notation_selection_circle(previous_state_idx // 2 + 1, previous_state_idx % 2)
                self.draw_notation_grid_onto_img(previous_state_idx)

                if previous_state_idx != self.move_count and not self.back_button_drawn:
                    self.draw_board_status('Viewing previous moves', text_color=(255, 255, 255))
                    self.draw_back_button()
                elif previous_state_idx == self.move_count and self.back_button_drawn:
                    self.set_current_game_state()
                    if self.move_count % 2 == 0:
                        self.draw_board_status(self.status_bar_current_text, text_color=(255, 255, 255))
                    else:
                        self.draw_board_status(self.status_bar_current_text, text_color=self.square_color_black)
            self.previous_key_time = time.perf_counter()

    def set_previous_game_state(self, previous_state_idx):
        self.current_board_state = self.previous_board_states[self.move_count, 0:].copy()

        previous_board_state, previous_draw_values = self.previous_board_states[previous_state_idx], self.previous_board_state_moves[previous_state_idx]
        all_draw_idxs = np.indices((8, 8))
        self.draw_board(np.column_stack((all_draw_idxs[1].flatten(), all_draw_idxs[0].flatten())), previous_board_state)
        if np.any(previous_draw_values) != 0:
            self.draw_board(previous_draw_values, previous_board_state, pre_move=True)
        self.previous_position_view_idx = previous_state_idx

    def return_to_previous_game_state(self):
        # To be removed - viewing previous notation is now handled using arrow keys - function remains temporarily
        pass

    def set_current_game_state(self):
        all_draw_idxs = np.indices((8, 8))
        self.draw_board(np.column_stack((all_draw_idxs[1].flatten(), all_draw_idxs[0].flatten())), self.current_board_state)
        if self.drawn_previous_moves is not None:
            self.draw_board(self.drawn_previous_moves, self.current_board_state, pre_move=True)
        elif self.drawn_moves is not None:
            self.draw_board(self.drawn_moves, self.current_board_state, pre_move=True)
        self.draw_notation_selection_circle(self.move_count // 2 + 1, self.move_count % 2)
        self.draw_notation_grid_onto_img()
        self.draw_back_button(undraw=True)

        self.previous_position_view_idx = None
        self.current_board_state = None
        self.notation_current_move_count = self.move_count

    def draw_menu(self, img):
        self.img[self.menu_bbox[0]:self.menu_bbox[1], self.menu_bbox[2]:self.menu_bbox[3]] = img

    def draw_initial_options_checkbox_and_return_bbox(self, text, img, vertical_placement_percentage, is_checked=False) -> np.ndarray:
        sound_text_size = cv2.getTextSize(text, self.font_face, self.font_scale_thickness[0], self.font_scale_thickness[1])
        xy_text_offset = int((img.shape[0] // 2 - sound_text_size[0][1]) * vertical_placement_percentage), int((img.shape[1] - sound_text_size[0][0]) * vertical_placement_percentage)
        xy_checkbox_offset = (img.shape[0] // 2 - self.checkbox_size_all_fill[0]) // 4, (img.shape[1] - sound_text_size[0][0]) // 4
        cv2.putText(img, text, (0 + xy_text_offset[1], 0 + xy_text_offset[0]), self.font_face, self.font_scale_thickness[0], (255, 255, 255), self.font_scale_thickness[1], cv2.LINE_AA)
        y_mp_checkbox = xy_text_offset[0] - self.checkbox_size_all_fill[0] // 2
        bbox_checkbox = np.array([sound_text_size[0][1] // 2 + y_mp_checkbox, sound_text_size[0][1] // 2 + y_mp_checkbox + self.checkbox_size_all_fill[0] // 2, img.shape[1] // 2 + xy_checkbox_offset[0], img.shape[1] // 2 + xy_checkbox_offset[0] + self.checkbox_size_all_fill[0] // 2],
                                 dtype=np.uint16)
        img[bbox_checkbox[0]:bbox_checkbox[1], bbox_checkbox[2]:bbox_checkbox[3]] = (255, 255, 255)

        if is_checked:
            self.fill_checkbox(img, bbox_checkbox, uncheck=False)

        return bbox_checkbox

    def draw_border(self, img, bbox):
        rectangle_border_points = (bbox[2] - self.border_thickness, bbox[0] - self.border_thickness), \
                                  (bbox[3] + self.border_thickness, bbox[1] + self.border_thickness)
        cv2.rectangle(img, rectangle_border_points[0], rectangle_border_points[1], (255, 255, 255), self.border_thickness, cv2.LINE_AA)

    def export_moves(self):
        if len(self.notation_tracker[0]) != 0:
            time_now = datetime.datetime.now()
            export_txt_file = f'{dir.export_directory}/{time_now.strftime("%d%m%y_%H%M%S")}.pgn'
            move_str = ''

            with open(export_txt_file, 'w') as f:
                f.write(f'[FEN "{self.fen_notation}"] \n')
                for i, white_move in enumerate(self.notation_tracker[0]):
                    move_str = f'{move_str}{str(i + 1)}. {white_move}'
                    try:
                        black_move = self.notation_tracker[1][i]
                        move_str = f'{move_str} {black_move} '
                    except IndexError:
                        break
                f.write(move_str)

    def fill_checkbox(self, img, bbox, uncheck=True):
        img[bbox[0]:bbox[1], bbox[2]:bbox[3]] = (255, 255, 255)
        if not uncheck:
            mp_xy = int((bbox[3] + bbox[2]) // 2), int((bbox[1] + bbox[0]) // 2)
            cv2.circle(img, mp_xy, self.checkbox_size_all_fill[1], (int(self.square_color_black[0]), int(self.square_color_black[1]), int(self.square_color_black[2])), -1, lineType=cv2.LINE_AA)

    def click_is_on(self, location, mouse_x, mouse_y):
        click_bool = False

        if location == 'board':
            bbox = self.board_img_bbox
        elif location == 'button':
            bbox = self.button_ranges
        elif location == 'menu':
            bbox = self.menu_bbox

        if len(bbox.shape) == 1:
            if bbox[2] < mouse_x < bbox[3] and bbox[0] < mouse_y < bbox[1]:
                click_bool = True
        else:
            if np.any(np.logical_and(np.logical_and(bbox[:, 0] <= mouse_y, bbox[:, 1] >= mouse_y), np.logical_and(bbox[:, 2] <= mouse_x, bbox[:, 3] >= mouse_x))):
                click_bool = True

        return click_bool

    def draw_board_status(self, text, text_color):
        status_bar_bbox = np.array([0, self.grid_square_size_yx[0], self.menu_bbox[2], self.menu_bbox[3]], dtype=np.uint16)
        self.draw_centered_text(self.img, text, bbox=status_bar_bbox, text_color=text_color, fill_color=(0, 0, 0))


white_i, black_i = 0, 1
non_capture_draw_i, capture_draw_i = 0, 1

# White, white_normal, white_valid_move
# Black, black_normal, black_valid_move


piece_selected = False


class Pieces(Enum):
    knight = 0
    bishop = 1
    rook = 2
    queen = 3
    king = 4
    pawn = 5
    pawn_movement = 5
    pawn_attack = 6


def return_non_pawn_movement_vectors_and_magnitudes():
    # All vectors, per numpy convention , are y, x movements
    orthogonal_vectors = np.array([[-1, 0],
                                   [1, 0],
                                   [0, 1],
                                   [0, -1]], dtype=np.int8)
    diagonal_vectors = np.array([[-1, 1],
                                 [-1, -1],
                                 [1, -1],
                                 [1, 1]], dtype=np.int8)
    knight_vectors = np.array([[-2, -1],
                               [-2, 1],
                               [2, -1],
                               [2, 1],
                               [-1, -2],
                               [1, -2],
                               [-1, 2],
                               [1, 2]], dtype=np.int8)

    all_vectors = np.vstack((orthogonal_vectors, diagonal_vectors, knight_vectors))
    vector_views = []
    vector_magnitudes = np.array([2, 9, 9, 9, 2, 2, 2], dtype=np.uint8)

    # Knight
    vector_views.append(all_vectors[orthogonal_vectors.shape[0] + diagonal_vectors.shape[0]:all_vectors.shape[0]])
    # Bishop
    vector_views.append(all_vectors[orthogonal_vectors.shape[0]:orthogonal_vectors.shape[0] + diagonal_vectors.shape[0]])
    # Rook
    vector_views.append(all_vectors[0:orthogonal_vectors.shape[0]])
    # Queen
    vector_views.append(all_vectors[0:orthogonal_vectors.shape[0] + diagonal_vectors.shape[0]])
    # King
    vector_views.append(all_vectors[0:orthogonal_vectors.shape[0] + diagonal_vectors.shape[0]])
    # Pawn_Movement_White
    vector_views.append(np.array([all_vectors[0]]))
    # Pawn_Attack_White
    vector_views.append(diagonal_vectors[0:2])

    return vector_views, vector_magnitudes


def return_white_squares_grid():
    chess_board_square_indicies = np.sum(np.indices((8, 8), dtype=np.uint8), axis=0)
    chess_board_white_squares_idxs = np.argwhere(chess_board_square_indicies % 2 == 0)
    chess_board_white_squares = np.full_like(chess_board_square_indicies, fill_value=black_i, dtype=np.uint0)
    chess_board_white_squares[chess_board_white_squares_idxs[:, 0], chess_board_white_squares_idxs[:, 1]] = white_i
    return chess_board_white_squares


def main():
    global new_game

    def set_starting_board_state(fisher=False):
        global game_has_not_ended, turn_i_current_opponent

        nonlocal rook_king_rook_has_not_moved, en_passant_tracker, king_closest_obstructing_pieces_is, board_state_pieces_colors_squares, \
            king_in_check_valid_piece_idxs, king_in_check_valid_moves, king_positions_yx, fisher_rook_xs

        if not fisher:
            non_pawn_row_values = np.array([Pieces.rook.value, Pieces.knight.value, Pieces.bishop.value, Pieces.queen.value, Pieces.king.value, Pieces.bishop.value, Pieces.knight.value, Pieces.rook.value], dtype=np.uint8) + 1
            fisher_rook_xs = None
            king_positions_yx = np.array([[7, 4], [0, 4]], dtype=np.int8)
        else:
            non_pawn_row_values = np.zeros(8, dtype=np.uint8)
            for bishop_i in (np.random.choice(np.array([1, 3, 5, 7])), np.random.choice(np.array([0, 2, 4, 6]))):
                non_pawn_row_values[bishop_i] = Pieces.bishop.value + 1
            queen_i = np.random.choice(np.argwhere(non_pawn_row_values == 0).flatten())
            non_pawn_row_values[queen_i] = Pieces.queen.value + 1
            for knight_i in np.random.choice(np.argwhere(non_pawn_row_values == 0).flatten(), 2, replace=False):
                non_pawn_row_values[knight_i] = Pieces.knight.value + 1
            remaining_squares = np.argwhere(non_pawn_row_values == 0)
            fisher_rook_xs = remaining_squares[0::2]
            non_pawn_row_values[fisher_rook_xs] = Pieces.rook.value + 1
            non_pawn_row_values[remaining_squares[1]] = Pieces.king.value + 1
            king_positions_yx = np.array([[7, remaining_squares[1]], [0, remaining_squares[1]]], dtype=np.int8)

        board_state_pieces_colors_squares[0:2, 0:] = 0
        for piece_color_i, pawn_row_i, non_pawn_row_i, in zip((white_i, black_i), (6, 1), (7, 0)):
            for row_i, row_values in zip((pawn_row_i, non_pawn_row_i), (Pieces.pawn.value + 1, non_pawn_row_values[0:])):
                y_board_idxs, x_board_idxs = np.full(8, row_i), np.arange(0, 8)
                board_state_pieces_colors_squares[0, y_board_idxs, x_board_idxs] = row_values
                board_state_pieces_colors_squares[1, y_board_idxs, x_board_idxs] = piece_color_i
                o_img.draw_board(np.column_stack((y_board_idxs, x_board_idxs)), board_state_pieces_colors_squares)

        y_empty_square_idxs = np.indices((8, 8))
        o_img.draw_board((np.column_stack((y_empty_square_idxs[0, 2:6].flatten(), y_empty_square_idxs[1, 2:6].flatten()))), board_state_pieces_colors_squares)
        o_img.previous_board_states[0, 0:] = board_state_pieces_colors_squares[0:]

        for i in (white_i, black_i):
            update_king_obstruction_array(king_positions_yx[i], king_closest_obstructing_pieces_is[i])

        # Clears move tracking arrays
        rook_king_rook_has_not_moved, en_passant_tracker = np.ones_like(rook_king_rook_has_not_moved), np.zeros_like(en_passant_tracker)
        king_in_check_valid_piece_idxs, king_in_check_valid_moves = [None, None], [[], []]
        turn_i_current_opponent = (white_i, black_i)

        o_img.set_fen_notation_str(row_values=non_pawn_row_values)
        o_img.status_bar_current_text = f'{o_img.status_bar_strs[0]} to move'
        o_img.draw_board_status(o_img.status_bar_current_text, text_color=(255, 255, 255))

        o_img.notation_tracker = [[], []]
        o_img.move_count = o_img.notation_current_move_count = 0
        game_has_not_ended = True

    def square_is_protected(potential_protection_vectors, potential_piece_ids, magnitude_i) -> bool:
        # Orthogonal, Diagonal, Knight
        # potential_protection_vectors = np.vstack((potential_protection_vectors, potential_protection_vectors))

        vector_types = np.argwhere(np.any(potential_protection_vectors == 0, axis=1)).flatten(), \
                       np.argwhere((np.sum(np.abs(potential_protection_vectors), axis=1) % 2 == 0)).flatten(), \
                       np.argwhere((np.sum(np.abs(potential_protection_vectors) % 2, axis=1) % 2 == 1)).flatten()
        for i, vector_type in enumerate(vector_types):
            if vector_type.shape[0] != 0:
                for vector_type_i_idx in vector_type:
                    piece_id = potential_piece_ids[vector_type_i_idx]
                    if magnitude_i == 1:
                        if i == 0:
                            accepted_piece_ids = np.hstack((obstructing_piece_identities[i], Pieces.king.value))
                        elif i == 1:
                            accepted_piece_ids = np.hstack((obstructing_piece_identities[i], Pieces.pawn.value, Pieces.king.value))
                        else:
                            accepted_piece_ids = obstructing_piece_identities[i]
                    else:
                        accepted_piece_ids = obstructing_piece_identities[i]

                    if piece_id in accepted_piece_ids:
                        return True

        return False

    def update_king_obstruction_array(king_position_yx, king_obstruction_array):
        obstructing_idxs = return_potential_moves(king_position_yx, (Pieces.queen.value,), check_type='obstruction')
        if obstructing_idxs is not None:
            obstructing_signs = np.sign(obstructing_idxs - (king_position_yx[0], king_position_yx[1]))
            obstructing_signs_tt = movement_vectors[Pieces.king.value][:, None] == obstructing_signs
            valid_idxs = np.argwhere(np.all(obstructing_signs_tt, axis=-1))
            invalid_idxs = np.invert((obstructing_signs_tt).all(-1).any(-1))
            king_obstruction_array[valid_idxs[:, 0]] = obstructing_idxs[valid_idxs[:, 1]]
            king_obstruction_array[invalid_idxs] = (8, 8)

    def return_vector_ranges(start_yx, end_yx, orthogonal):
        if orthogonal:
            obstruction_vector = end_yx - start_yx
            obstruction_idx = np.argwhere(obstruction_vector != 0).flatten()[0]
            # Magnitude is not 1
            if end_yx[obstruction_idx] > start_yx[obstruction_idx]:
                variable_obstruction_idxs = np.arange(start_yx[obstruction_idx], end_yx[obstruction_idx])
            else:
                variable_obstruction_idxs = np.arange(end_yx[obstruction_idx], start_yx[obstruction_idx])

            if obstruction_idx == 0:
                compare_idxs_yx = np.column_stack((variable_obstruction_idxs, np.full_like(variable_obstruction_idxs, start_yx[1])))
            else:
                compare_idxs_yx = np.column_stack((np.full_like(variable_obstruction_idxs, start_yx[0]), variable_obstruction_idxs))
            return compare_idxs_yx
        # Diagonal check
        else:
            vector = end_yx - start_yx
            vector_sign = np.sign(vector)
            vector_magnitude_range = np.column_stack((np.arange(1, abs(vector[0]) + 1), np.arange(1, abs(vector[0]) + 1)))
            return start_yx + vector_magnitude_range * vector_sign

    def return_potential_moves(end_yx: tuple, piece_ids: tuple, check_type: str = 'moves', pawn_movement=False, pawn_attack=False, en_passant_check=False, obstruction_check=False, vector_magnitude_overwrite=(None, None)) -> np.ndarray or None or False:
        valid_squares_yx = []
        if piece_ids[0] == Pieces.pawn.value:
            piece_ids = (Pieces.pawn_movement.value, Pieces.pawn_attack.value)

        for i, piece_id in enumerate(piece_ids):
            if vector_magnitude_overwrite[0] is None:
                vectors, magnitude = movement_vectors[piece_id], movement_magnitudes[piece_id]
                if piece_id == Pieces.pawn_movement.value:
                    pawn_movement, pawn_attack = True, False
                    if turn_i_current_opponent[0] == white_i:
                        if end_yx[0] == 6:
                            magnitude += 1
                    else:
                        vectors = vectors.copy() * -1
                        if end_yx[0] == 1:
                            magnitude += 1

                elif piece_id == Pieces.pawn_attack.value:
                    vectors, magnitude = movement_vectors[Pieces.pawn_attack.value], movement_magnitudes[Pieces.pawn_attack.value]
                    pawn_attack, pawn_movement = True, False
                    if turn_i_current_opponent[0] == white_i:
                        if end_yx[0] == 3:
                            en_passant_check = True
                    else:
                        vectors = vectors.copy() * -1
                        if end_yx[0] == 4:
                            en_passant_check = True
                else:
                    vectors, magnitude = movement_vectors[piece_id], movement_magnitudes[piece_id]
            else:
                vectors, magnitude = vector_magnitude_overwrite[0][i], vector_magnitude_overwrite[1][i]

            if obstruction_check:
                if np.any(np.all(king_closest_obstructing_pieces_is[turn_i_current_opponent[0]] == end_yx, axis=1)):
                    obstruction_vector_i = np.argwhere((np.all(king_closest_obstructing_pieces_is[turn_i_current_opponent[0]] == end_yx, axis=1))).flatten()[0]
                    next_piece_yx = return_potential_moves(end_yx, (0,), vector_magnitude_overwrite=((np.array([movement_vectors[Pieces.king.value][obstruction_vector_i]]),), (9,)), check_type='obstruction')
                    if next_piece_yx is not None:
                        next_piece_properties = board_state_pieces_colors_squares[0:2, next_piece_yx[:, 0], next_piece_yx[:, 1]]
                        potential_protection_vector = np.array([movement_vectors[Pieces.king.value][obstruction_vector_i]])
                        if np.any(potential_protection_vector == 0) and next_piece_properties[0, 0] - 1 in obstructing_piece_identities[0] or abs(potential_protection_vector[0, 0]) == abs(potential_protection_vector[0, 1]) and next_piece_properties[0, 0] - 1 in obstructing_piece_identities[1]:
                            if next_piece_properties[1] == turn_i_current_opponent[1]:
                                if np.any(np.all(vectors == potential_protection_vector, axis=1)):
                                    vectors = np.vstack((potential_protection_vector, potential_protection_vector * -1))
                                else:
                                    magnitude = 1

                                '''magnitude_i = np.max(np.abs(next_piece_yx - king_positions_yx[turn_i_current_opponent[0]]))
                                if square_is_protected(potential_protection_vector, (next_piece_properties[0]), magnitude_i):
                                    if np.any(np.all(vectors == potential_protection_vector, axis=1)):
                                        vectors = potential_protection_vector
                                    else:
                                        magnitude = 1'''

                        # If the next piece is a valid piece protecting it, the chosen piece can only move on that given vector

                        # Otherwise, the next piece is a piece of the same side, and set that as the potential next obstructing piece
                        '''else:
                            o_img.obstruction_next_vector_sign, obstruction_next_vector_i, o_img.obstruction_next_piece_yx = np.sign(potential_protection_vector), obstruction_vector_i, next_piece_yx
                            king_closest_obstructing_pieces_is[current_turn_i, obstruction_vector_i] = next_piece_yx[0]
                    else:
                        king_closest_obstructing_pieces_is[current_turn_i, obstruction_vector_i] = (8, 8)'''

            valid_vectors = np.ones(vectors.shape[0], dtype=np.uint0)
            for magnitude_i in range(1, magnitude):
                valid_vector_idxs = np.argwhere(valid_vectors).flatten()

                potential_moves = end_yx + (vectors[valid_vector_idxs]) * magnitude_i
                valid_squares = np.all(np.logical_and(potential_moves >= 0, potential_moves < 8), axis=1)
                valid_square_is = np.argwhere(valid_squares).flatten()
                valid_vectors[valid_vector_idxs[np.invert(valid_squares)]] = 0

                valid_potential_moves = potential_moves[valid_square_is]
                valid_vector_idxs = valid_vector_idxs[valid_square_is]
                piece_identities_colors = board_state_pieces_colors_squares[0:2, valid_potential_moves[:, 0], valid_potential_moves[:, 1]]
                valid_pieces = np.argwhere(piece_identities_colors[0] != 0).flatten()

                if check_type == 'moves':
                    empty_squares = np.argwhere(piece_identities_colors[0] == 0).flatten()
                    if empty_squares.shape[0] != 0:
                        for empty_square_i in empty_squares:
                            if not pawn_attack:
                                valid_squares_yx.append(valid_potential_moves[empty_square_i])
                            elif en_passant_check:
                                if en_passant_tracker[turn_i_current_opponent[1], valid_potential_moves[empty_square_i, 1]]:
                                    valid_squares_yx.append(valid_potential_moves[empty_square_i])

                    if valid_pieces.shape[0] != 0:
                        if not pawn_movement:
                            valid_opponent_pieces = np.argwhere(piece_identities_colors[1, valid_pieces] == turn_i_current_opponent[1]).flatten()
                            if valid_opponent_pieces.shape[0] != 0:
                                valid_squares_yx.append(valid_potential_moves[valid_pieces[valid_opponent_pieces]])
                        valid_vectors[valid_vector_idxs[valid_pieces]] = 0
                else:
                    if valid_pieces.shape[0] != 0:
                        if check_type == 'obstruction':
                            valid_squares_yx.append(potential_moves[valid_square_is[valid_pieces]])
                        elif check_type == 'protection':
                            if valid_pieces.shape[0] != 0:
                                valid_ally_pieces = np.argwhere(piece_identities_colors[1, valid_pieces] == turn_i_current_opponent[1]).flatten()
                                if valid_ally_pieces.shape[0] != 0:
                                    potential_protection_moves = valid_potential_moves[valid_pieces[valid_ally_pieces]]
                                    # Vector types are Orthogonal, Diagonal, and Knight

                                    if len(potential_protection_moves.shape) == 1:
                                        potential_protection_moves = np.array([[potential_protection_moves]])

                                    potential_protection_piece_ids = board_state_pieces_colors_squares[0, potential_protection_moves[:, 0], potential_protection_moves[:, 1]] - 1
                                    potential_protection_vectors = vectors[valid_vector_idxs[valid_pieces[valid_ally_pieces]]]
                                    if square_is_protected(potential_protection_vectors, potential_protection_piece_ids, magnitude_i):
                                        return False
                            # If the protecting piece is not a king,

                        valid_vectors[valid_vector_idxs[valid_pieces]] = 0

                if np.all(valid_vectors == 0):
                    break

        if len(valid_squares_yx) != 0:
            if len(valid_squares_yx) == 1:
                if len(valid_squares_yx[0].shape) == 1:
                    valid_squares_yx = np.array([valid_squares_yx[0]])
                else:
                    valid_squares_yx = valid_squares_yx[0]
            else:
                valid_squares_yx = np.vstack(valid_squares_yx)
            return valid_squares_yx
        else:
            return None

    def draw_potential_moves(piece_yx_idx):
        nonlocal board_state_pieces_colors_squares, fisher_rook_xs
        valid_squares_yx = None
        if king_in_check_valid_piece_idxs[turn_i_current_opponent[0]] is None:
            piece_id = board_state_pieces_colors_squares[0, piece_yx_idx[0], piece_yx_idx[1]] - 1
            if board_state_pieces_colors_squares[1, piece_yx_idx[0], piece_yx_idx[1]] == turn_i_current_opponent[0] and piece_id != -1:
                if piece_id == Pieces.king.value:
                    # Castle Check
                    if king_in_check_valid_piece_idxs[turn_i_current_opponent[0]] is None:
                        if rook_king_rook_has_not_moved[turn_i_current_opponent[0], 1]:
                            # Fisher_rook_xs will be (2) array with the x values that they are located if a new game is started with o_img.fisher = True
                            if fisher_rook_xs is not None:
                                rook_x_idxs, offsets = fisher_rook_xs, fisher_rook_xs - king_positions_yx[turn_i_current_opponent[0], 1]
                            else:
                                rook_x_idxs, offsets = (0, 7), (-4, 3)

                            for rook_has_moved_i, rook_x_i, king_x_offset in zip((0, 2), rook_x_idxs, offsets):
                                if rook_king_rook_has_not_moved[turn_i_current_opponent[0], rook_has_moved_i]:
                                    able_to_castle = True
                                    #Fisher castle handling
                                    if fisher_rook_xs is not None:
                                        if rook_x_i > piece_yx_idx[1]:
                                            destination_squares_yx, rook_x_idx = np.column_stack((np.repeat(piece_yx_idx[0], 2), (5, 6))), fisher_rook_xs[1]
                                            castle_x_idxs = np.arange(piece_yx_idx[1] + 1, rook_x_i)
                                        else:
                                            destination_squares_yx, rook_x_idx = np.column_stack((np.repeat(piece_yx_idx[0], 2), (2, 3))), fisher_rook_xs[0]
                                            castle_x_idxs = np.arange(rook_x_i + 1, piece_yx_idx[1])

                                        destination_square_is_protected = False
                                        for destination_square in destination_squares_yx:
                                            square_is_not_protected = return_potential_moves((destination_square, ), (Pieces.queen.value, Pieces.knight.value), check_type='protection')
                                            if square_is_not_protected is not None:
                                                destination_square_is_protected = True
                                                break

                                        if destination_square_is_protected:
                                            continue

                                        board_state = board_state_pieces_colors_squares[0:2, destination_squares_yx[:, 0], destination_squares_yx[:, 1]]
                                        occupied_squares = np.argwhere(board_state[0] > 0).flatten()
                                        #If squares are occupied, checks to see if the squares are occupied by the king/respective rook, if not, continue the loop
                                        if len(occupied_squares) != 0:
                                            board_state, destination_squares = board_state[:, occupied_squares], destination_squares_yx[occupied_squares]
                                            if len(occupied_squares) == 1:
                                                valid_rook = np.logical_and(np.logical_and(board_state[0] == Pieces.rook.value + 1, destination_squares_yx[:, 1] == rook_x_idx), board_state[1] == turn_i_current_opponent[0])
                                                valid_king = np.logical_and(board_state[0, 0] == Pieces.king.value + 1, board_state[1] == turn_i_current_opponent[0])
                                            else:
                                                valid_rook = np.logical_and(np.logical_and(board_state[0] == Pieces.rook.value + 1, destination_squares_yx[:, 1] == rook_x_idx), board_state[1] == turn_i_current_opponent[0])
                                                valid_king = np.logical_and(board_state[0] == Pieces.king.value + 1, board_state[1] == turn_i_current_opponent[0])


                                            if np.count_nonzero(valid_rook) + np.count_nonzero(valid_king) != len(occupied_squares):
                                                continue


                                    elif rook_x_i > piece_yx_idx[1]:
                                        castle_x_idxs = np.arange(piece_yx_idx[1] + 1, rook_x_i)
                                    else:
                                        castle_x_idxs = np.arange(rook_x_i + 1, piece_yx_idx[1])


                                    #Castle_x_idxs will always be nonzero for normal games, but fisher can create a scenario where the king and rook are adjacent but can still castle.
                                    #In that instance, set able to castle to true, as the fisher check is done in the code block if fisher_rook_xs is not None:
                                    if castle_x_idxs.shape[0] != 0:
                                        castle_idxs_all = np.column_stack((np.full_like(castle_x_idxs, piece_yx_idx[0]), castle_x_idxs))
                                        if np.all(board_state_pieces_colors_squares[0, castle_idxs_all[:, 0], castle_idxs_all[:, 1]] == 0):
                                            for castle_idx in castle_idxs_all:
                                                square_is_not_protected = return_potential_moves(castle_idx, (Pieces.queen.value, Pieces.knight.value), check_type='protection')
                                                if square_is_not_protected is not None:
                                                    able_to_castle = False
                                                    break
                                        else:
                                            able_to_castle = False
                                    else:
                                        able_to_castle = True

                                    if able_to_castle:
                                        if valid_squares_yx is not None:
                                            valid_squares_yx = np.vstack((valid_squares_yx, np.array([piece_yx_idx[0], piece_yx_idx[1] + king_x_offset], dtype=np.uint8)))
                                        else:
                                            valid_squares_yx = np.array([piece_yx_idx[0], piece_yx_idx[1] + king_x_offset], dtype=np.uint8)


                                #Checks for Non Castle Moves
                                single_moves = return_potential_moves(piece_yx_idx, (Pieces.king.value,))
                                if single_moves is not None:
                                    valid_single_moves = np.ones(single_moves.shape[0], dtype=np.uint0)
                                    for i, single_move in enumerate(single_moves):
                                        square_is_not_protected = return_potential_moves(single_move, (Pieces.queen.value, Pieces.knight.value), check_type='protection')
                                        if square_is_not_protected is not None:
                                            valid_single_moves[i] = 0
                                    valid_non_castle_moves = single_moves[np.argwhere(valid_single_moves).flatten()]

                                    if valid_non_castle_moves.shape[0] != 0:
                                        if valid_squares_yx is not None:
                                            valid_squares_yx = np.vstack((valid_squares_yx, valid_non_castle_moves))
                                        else:
                                            valid_squares_yx = valid_non_castle_moves



                else:
                    valid_squares_yx = return_potential_moves(piece_yx_idx, (piece_id,), obstruction_check=True)

        else:
            valid_piece_to_prevent_check = np.all(king_in_check_valid_piece_idxs[turn_i_current_opponent[0]] == piece_yx_idx, axis=1)
            if valid_piece_to_prevent_check.any():
                valid_piece_i = np.argwhere(valid_piece_to_prevent_check).flatten()
                valid_squares_yx = king_in_check_valid_moves[int(turn_i_current_opponent[0])][int(valid_piece_i)]

                # Pawn Handling

            # Draws Potential moves if available
        if valid_squares_yx is not None:
            if o_img.drawn_previous_moves is not None:
                o_img.draw_board(o_img.drawn_previous_moves, board_state_pieces_colors_squares)
            o_img.drawn_previous_moves = None

            o_img.draw_board(np.vstack((valid_squares_yx, piece_yx_idx)), board_state_pieces_colors_squares, pre_move=True)
            o_img.piece_idx_selected = piece_yx_idx
            o_img.drawn_moves = np.vstack((valid_squares_yx, piece_yx_idx))
        else:
            o_img.piece_idx_selected = o_img.drawn_moves = None

            # o_img.draw_board(np.array([piece_yx_idx], dtype=np.uint16), board_state_pieces_colors_squares, pre_move=True)

    def check_for_valid_moves(piece_id, selected_yx, checking_piece_yx, checking_piece_vector) -> (bool, bool):
        global turn_i_current_opponent
        in_check, no_valid_moves = False, False
        checking_piece_idxs, checking_obstruction_squares = opponent_is_in_check(piece_id, selected_yx, checking_piece_yx, checking_piece_vector)
        turn_i_current_opponent = (turn_i_current_opponent[1], turn_i_current_opponent[0])

        piece_moves = np.argwhere(np.logical_and(np.logical_and(board_state_pieces_colors_squares[0] != 0, board_state_pieces_colors_squares[0] != Pieces.king.value + 1),
                                                 board_state_pieces_colors_squares[1] == turn_i_current_opponent[0]))
        piece_ids = board_state_pieces_colors_squares[[0], piece_moves[:, 0], piece_moves[:, 1]]

        if len(checking_piece_idxs) != 0:
            in_check = True
            potential_king_moves = return_potential_moves(king_positions_yx[turn_i_current_opponent[0]], (Pieces.king.value,))
            checking_piece_vectors = king_positions_yx[turn_i_current_opponent[0]] - checking_piece_idxs[0]
            if potential_king_moves is not None:
                valid_king_moves = np.ones(potential_king_moves.shape[0], dtype=np.uint0)
                for i, single_move in enumerate(potential_king_moves):
                    king_movement_vector = single_move - checking_piece_idxs[0]
                    # Below calculates if movement is on the same vector as the checking vector, if it is it is an invalid move (if it is not capturing the piece)
                    if np.any(single_move != checking_piece_idxs[0], axis=-1) and (abs(king_movement_vector[0]) == abs(king_movement_vector[1]) or np.any(king_movement_vector == 0)):
                        if abs(king_movement_vector[0]) == abs(king_movement_vector[1]) or np.any(king_movement_vector == 0):
                            if np.all(np.sign(king_movement_vector) == np.sign(checking_piece_vectors), axis=-1):
                                valid_king_moves[i] = 0
                                continue
                    square_is_not_protected = return_potential_moves(single_move, (Pieces.queen.value, Pieces.knight.value), check_type='protection')
                    if square_is_not_protected is not None:
                        valid_king_moves[i] = 0

                if valid_king_moves.any():
                    king_in_check_valid_piece_idxs[turn_i_current_opponent[0]] = np.array([king_positions_yx[turn_i_current_opponent[0]]])
                    king_in_check_valid_moves[turn_i_current_opponent[0]].append(potential_king_moves[np.argwhere(valid_king_moves).flatten()])

            if len(checking_piece_idxs) != 2:
                for piece_id, piece_yx in zip(piece_ids, piece_moves):
                    potential_moves = return_potential_moves(piece_yx, (piece_id - 1,), obstruction_check=True)
                    if potential_moves is not None:
                        valid_moves_in_checking_idxs = (potential_moves[:, None] == checking_obstruction_squares).all(-1).any(-1)
                        if np.any(valid_moves_in_checking_idxs):
                            valid_moves = potential_moves[valid_moves_in_checking_idxs]
                            if king_in_check_valid_piece_idxs[turn_i_current_opponent[0]] is None:
                                king_in_check_valid_piece_idxs[turn_i_current_opponent[0]] = np.array([piece_yx], dtype=np.uint8)
                            else:
                                king_in_check_valid_piece_idxs[turn_i_current_opponent[0]] = np.vstack((king_in_check_valid_piece_idxs[turn_i_current_opponent[0]], piece_yx))
                            king_in_check_valid_moves[turn_i_current_opponent[0]].append(valid_moves)
            if king_in_check_valid_piece_idxs[turn_i_current_opponent[0]] is None:
                no_valid_moves = True
        else:
            # Stalemate Check
            for piece_id, piece_yx in zip(piece_ids, piece_moves):
                potential_moves = return_potential_moves(piece_yx, (piece_id - 1,), obstruction_check=True)
                if potential_moves is not None:
                    no_valid_moves = False
                break

        return in_check, no_valid_moves

    def opponent_is_in_check(piece_id, selected_yx, checking_piece_yx, checking_piece_vector) -> (list, list):
        checking_pieces_idxs, checking_squares = [], []
        opposing_king_yx = king_positions_yx[turn_i_current_opponent[1]]

        # Checks for a check caused by the piece's movement clearing an attack to the king
        if piece_id != Pieces.king.value:
            if checking_piece_yx is not None:
                checking_piece_id = board_state_pieces_colors_squares[0, checking_piece_yx[0], checking_piece_yx[1]] - 1
                if np.any(checking_piece_vector == 0):
                    if checking_piece_id in obstructing_piece_identities[0]:
                        checking_pieces_idxs.append(checking_piece_yx)
                        if np.any(np.abs(checking_piece_yx - opposing_king_yx) >= 2):
                            checking_squares.append(return_vector_ranges(opposing_king_yx, checking_piece_yx, orthogonal=True))
                elif checking_piece_id in obstructing_piece_identities[1]:
                    checking_pieces_idxs.append(checking_piece_yx)
                    if np.any(np.abs(checking_piece_yx - opposing_king_yx) >= 2):
                        checking_squares.append(return_vector_ranges(opposing_king_yx, checking_piece_yx, orthogonal=False))

        # Checks for a check caused by the actual piece's attack vector
        check_moves = return_potential_moves(selected_yx, (piece_id,))
        if check_moves is not None:
            if np.all(opposing_king_yx == check_moves, axis=1).any():
                checking_pieces_idxs.append(selected_yx)
                if len(checking_pieces_idxs) != 0:
                    if piece_id != Pieces.knight.value and Pieces != Pieces.pawn.value:
                        if np.any(np.abs(opposing_king_yx - selected_yx) >= 2):
                            if selected_yx[0] - opposing_king_yx[0] == 0 or selected_yx[1] - opposing_king_yx[1] == 0:
                                vector_ranges = return_vector_ranges(opposing_king_yx, selected_yx, orthogonal=True)
                            else:
                                vector_ranges = return_vector_ranges(opposing_king_yx, selected_yx, orthogonal=False)
                            checking_squares = vector_ranges
                        else:
                            checking_squares = np.array([selected_yx])
                    else:
                        checking_squares = np.array([selected_yx])

        return checking_pieces_idxs, checking_squares

    def draw_selected_move(piece_next_yx, output_img_loc: OutputImg):
        global game_has_not_ended
        turn_i = turn_i_current_opponent[0]
        piece_initial_yx, drawn_idxs = output_img_loc.piece_idx_selected, output_img_loc.drawn_moves
        drawn_idx_compare = np.all(drawn_idxs == piece_next_yx, axis=1)
        in_drawn_idxs = np.argwhere(drawn_idx_compare).flatten()
        if in_drawn_idxs.shape[0] != 0 and np.any(piece_next_yx != piece_initial_yx, axis=-1):
            draw_idxs = np.vstack((piece_initial_yx, piece_next_yx))
            piece_id = board_state_pieces_colors_squares[0, piece_initial_yx[0], piece_initial_yx[1]] - 1
            piece_draw, piece_promoted, captured_piece, castle_text, game_state_text = Pieces(piece_id), None, False, None, None
            if piece_id == Pieces.rook.value:
                rook_castle_y_check = 7 if turn_i == 0 else 0
                if piece_initial_yx[0] == rook_castle_y_check:
                    if fisher_rook_xs is None:
                        rook_movement_tracker_i = 0 if piece_initial_yx[1] == 0 else 2
                    else:
                        rook_movement_tracker_i = 0 if piece_initial_yx[1] == fisher_rook_xs[0] else 2
                    rook_king_rook_has_not_moved[turn_i, rook_movement_tracker_i] = 0

            elif piece_id == Pieces.king.value:
                selected_piece_id_color = board_state_pieces_colors_squares[0:2, piece_next_yx[0], piece_next_yx[1]]
                # If selected piece is a rook and selected piece is the same color as the king, it is a castle
                if selected_piece_id_color[0] == Pieces.rook.value + 1 and selected_piece_id_color[1] == turn_i_current_opponent[0]:
                    # Kingside
                    if piece_next_yx[1] - piece_initial_yx[1] > 0:
                        castle_text = '0-0'
                        rook_idx, king_idx = (piece_initial_yx[0], 5), (piece_initial_yx[0], 6)
                    # Queenside
                    else:
                        castle_text = '0-0-0'
                        rook_idx, king_idx = (piece_initial_yx[0], 3), (piece_initial_yx[0], 2)

                    for piece_idxs in (piece_initial_yx, piece_next_yx):
                        board_state_pieces_colors_squares[0:2, piece_idxs[0], piece_idxs[1]] = (0, 0)

                    for piece_i, piece_id_loc in zip((rook_idx, king_idx), (Pieces.rook.value + 1, Pieces.king.value + 1)):
                        board_state_pieces_colors_squares[0:2, piece_i[0], piece_i[1]] = (piece_id_loc, turn_i)

                    draw_idxs = np.vstack((piece_initial_yx, piece_next_yx, rook_idx, king_idx))

                    rook_king_rook_has_not_moved[turn_i_current_opponent[0], 1] = 0
                    king_positions_yx[turn_i_current_opponent[0]] = king_idx

            elif piece_id == Pieces.pawn.value:
                if abs(piece_initial_yx[0] - piece_next_yx[0]) == 2:
                    en_passant_tracker[turn_i_current_opponent[0], piece_initial_yx[1]] = 1
                # En Passant Capture, no piece present, set the board state capture
                else:
                    # Capture made, check for en passant draw necessity
                    if piece_next_yx[1] - piece_initial_yx[1] != 0:
                        # Black Capture
                        if piece_next_yx[0] - piece_initial_yx[0] == 1:
                            # Black Capture En Passant
                            if board_state_pieces_colors_squares[0, piece_next_yx[0], piece_next_yx[1]] == 0:
                                board_state_pieces_colors_squares[0:2, piece_next_yx[0] - 1, piece_next_yx[1]] = (0, 0)
                                draw_idxs = np.vstack((draw_idxs, (piece_next_yx[0] - 1, piece_next_yx[1])))
                                captured_piece = True
                        # White Capture
                        else:
                            if board_state_pieces_colors_squares[0, piece_next_yx[0], piece_next_yx[1]] == 0:
                                board_state_pieces_colors_squares[0:2, piece_next_yx[0] + 1, piece_next_yx[1]] = (0, 0)
                                draw_idxs = np.vstack((draw_idxs, (piece_next_yx[0] + 1, piece_next_yx[1])))
                                captured_piece = True

            if board_state_pieces_colors_squares[0, piece_next_yx[0], piece_next_yx[1]] != 0:
                captured_piece = True

            potential_checking_piece_yx, potential_checking_piece_vector = None, None
            if piece_id != Pieces.king.value:
                board_state_pieces_colors_squares[0:2, piece_next_yx[0], piece_next_yx[1]] = board_state_pieces_colors_squares[0:2, piece_initial_yx[0], piece_initial_yx[1]]
                board_state_pieces_colors_squares[0:2, piece_initial_yx[0], piece_initial_yx[1]] = (0, 0)


                king_movement_vectors = movement_vectors[Pieces.king.value]
                for i, (obstruction_array, king_yx) in enumerate(zip(king_closest_obstructing_pieces_is, king_positions_yx)):
                    if np.any(np.all(obstruction_array == piece_initial_yx, axis=1)):
                        initial_vector = np.sign(piece_initial_yx - king_yx)
                        obstruction_i = np.argwhere(np.all(king_movement_vectors == initial_vector, axis=1)).flatten()
                        next_obstruction_idx = return_potential_moves(king_yx, (0,), check_type='obstruction', vector_magnitude_overwrite=((np.array([initial_vector]),), (9,)))
                        if next_obstruction_idx is not None:
                            next_obstruction_idx = next_obstruction_idx.flatten()
                            obstruction_array[obstruction_i] = next_obstruction_idx
                            if i == turn_i_current_opponent[1]:
                                if board_state_pieces_colors_squares[1, next_obstruction_idx[0], next_obstruction_idx[1]] != turn_i_current_opponent[i]:
                                    potential_checking_piece_yx, potential_checking_piece_vector = next_obstruction_idx, initial_vector

                    selected_piece_vector = piece_next_yx - king_yx
                    if np.any(selected_piece_vector == 0) or np.abs(selected_piece_vector[0]) == np.abs(selected_piece_vector[1]):
                        selected_piece_sign = np.sign(selected_piece_vector)
                        obstruction_i = np.argwhere(np.all(king_movement_vectors == selected_piece_sign, axis=1)).flatten()
                        next_obstruction_idx = return_potential_moves(king_yx, (0,), check_type='obstruction', vector_magnitude_overwrite=((np.array([selected_piece_sign]),), (9,)))
                        if next_obstruction_idx is not None:
                            obstruction_array[obstruction_i] = next_obstruction_idx

            output_img_loc.draw_board(draw_idxs, board_state_pieces_colors_squares)
            output_img_loc.draw_board(drawn_idxs, board_state_pieces_colors_squares)
            # Promotion Handler
            if piece_id == Pieces.pawn.value:
                if (turn_i_current_opponent[0] == black_i and piece_next_yx[0] == 7) or (turn_i_current_opponent[0] == white_i and piece_next_yx[0] == 0):
                    o_img.draw_promotion_selector(piece_next_yx[1], board_state_pieces_colors_squares, turn_i_current_opponent[0])

                    while o_img.promotion_ui_bbox_yx is not None:
                        cv2.imshow(o_img.window_name, o_img.img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            break
                    piece_id = board_state_pieces_colors_squares[0, piece_next_yx[0], piece_next_yx[1]] - 1
                    piece_promoted = Pieces(piece_id)

            update_king_obstruction_array(king_positions_yx[turn_i_current_opponent[0]], king_closest_obstructing_pieces_is[turn_i_current_opponent[0]])

            in_check, opponent_has_no_valid_moves = check_for_valid_moves(piece_id, piece_next_yx, potential_checking_piece_yx, potential_checking_piece_vector)
            if opponent_has_no_valid_moves:
                game_has_not_ended = False
                if in_check:
                    game_state_text = '++'
                    o_img.status_bar_current_text = f'{o_img.status_bar_strs[turn_i_current_opponent[1]]} has won!'
                    if turn_i_current_opponent[1] == white_i:
                        text_color = (255, 255, 255)
                    else:
                        text_color = o_img.square_color_black
                else:
                    game_state_text = '='
                    o_img.status_bar_current_text = f"Stalemate! You're both losers!"
                    text_color = (255, 255, 255)
            else:
                if turn_i_current_opponent[0] == white_i:
                    text_color = (255, 255, 255)
                else:
                    text_color = o_img.square_color_black
                if in_check:
                    game_state_text = '+'
                    o_img.status_bar_current_text = f'{o_img.status_bar_strs[turn_i_current_opponent[0]]} is in check!'
                else:
                    o_img.status_bar_current_text = f'{o_img.status_bar_strs[turn_i_current_opponent[0]]} to move'

            o_img.draw_board_status(o_img.status_bar_current_text, text_color)

            king_in_check_valid_piece_idxs[turn_i_current_opponent[1]] = None
            king_in_check_valid_moves[turn_i_current_opponent[1]].clear()
            output_img_loc.piece_idx_selected, output_img_loc.drawn_moves = None, None
            en_passant_tracker[turn_i_current_opponent[0]] = 0
            cv2.imshow(o_img.window_name, o_img.img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
            if o_img.sound_is_on:
                if captured_piece:
                    vlc.MediaPlayer(dir.capture_sound).play()
                else:
                    vlc.MediaPlayer(dir.move_sound).play()

            if o_img.drawn_previous_moves is not None:
                o_img.draw_board(o_img.drawn_previous_moves, board_state_pieces_colors_squares)
            o_img.drawn_previous_moves = np.vstack((piece_next_yx, piece_initial_yx))
            o_img.draw_board(o_img.drawn_previous_moves, board_state_pieces_colors_squares, pre_move=True)
            o_img.previous_board_states[o_img.move_count + 1] = board_state_pieces_colors_squares.copy()
            o_img.previous_board_state_moves[o_img.move_count + 1] = o_img.drawn_previous_moves.copy()

            potential_ambiguous_moves = return_potential_moves(piece_next_yx, (piece_draw.value,), check_type='moves', )
            if potential_ambiguous_moves is not None:
                color_ids = board_state_pieces_colors_squares[0:2, potential_ambiguous_moves[:, 0], potential_ambiguous_moves[:, 1]]
                same_piece_i = np.logical_and(color_ids[0] == piece_draw.value + 1, color_ids[1] == turn_i_current_opponent[0])
                if np.any(same_piece_i):
                    same_pieces = potential_ambiguous_moves[same_piece_i]
                    if same_pieces.shape[0] == 1:
                        if same_pieces[0, 0] == piece_initial_yx[0, 0]:
                            o_img.draw_notation_img(turn_i, piece_initial_yx, piece_next_yx, piece_draw, row_identifier=True, piece_captured=captured_piece, castle_text=castle_text, piece_promotion=piece_promoted, game_state_text=game_state_text)
                        else:
                            o_img.draw_notation_img(turn_i, piece_initial_yx, piece_next_yx, piece_draw, column_identifier=True, piece_captured=captured_piece, castle_text=castle_text, piece_promotion=piece_promoted, game_state_text=game_state_text)
                    else:
                        o_img.draw_notation_img(turn_i, piece_initial_yx, piece_next_yx, piece_draw, column_identifier=True, row_identifier=True, piece_captured=captured_piece, castle_text=castle_text, piece_promotion=piece_promoted, game_state_text=game_state_text)
                else:
                    o_img.draw_notation_img(turn_i, piece_initial_yx, piece_next_yx, piece_draw, piece_captured=captured_piece, piece_promotion=piece_promoted, castle_text=castle_text, game_state_text=game_state_text)
            else:
                o_img.draw_notation_img(turn_i, piece_initial_yx, piece_next_yx, piece_draw, piece_captured=captured_piece, piece_promotion=piece_promoted, castle_text=castle_text, game_state_text=castle_text)

        elif np.logical_and(board_state_pieces_colors_squares[0, piece_next_yx[0], piece_next_yx[1]] != 0,
                            board_state_pieces_colors_squares[1, piece_next_yx[0], piece_next_yx[1]] == turn_i_current_opponent[0]):
            output_img_loc.draw_board(drawn_idxs, board_state_pieces_colors_squares)
            draw_potential_moves(piece_next_yx)

    def mouse_handler(event, x, y, flags=None, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            if o_img.click_is_on('board', x, y):
                if o_img.current_menu == 'settings':
                    o_img.finalize_accent_color()
                elif o_img.current_menu == 'notation':
                    if o_img.notation_current_move_count != o_img.move_count:
                        o_img.set_current_game_state()
                        o_img.draw_back_button(undraw=True)
                elif o_img.current_menu == 'newgame':
                    o_img.current_menu = 'notation'
                    o_img.draw_notation_grid_onto_img(o_img.move_count)
                    o_img.draw_back_button(undraw=True)

                if game_has_not_ended:
                    idx_y_x = o_img.return_y_x_idx(x, y)
                    if o_img.promotion_ui_bbox_yx is not None:
                        if o_img.promotion_ui_bbox_yx[0] < y < o_img.promotion_ui_bbox_yx[1] and o_img.promotion_ui_bbox_yx[2] < x < o_img.promotion_ui_bbox_yx[3]:
                            selected_promotion_idx = ((y - o_img.promotion_ui_bbox_yx[0]) // o_img.grid_square_size_yx[0])
                            if turn_i_current_opponent[0] == white_i:
                                board_state_pieces_colors_squares[0:2, 0, idx_y_x[1]] = o_img.promotion_array_piece_idxs[turn_i_current_opponent[0], selected_promotion_idx], turn_i_current_opponent[0]
                                o_img.draw_board((np.column_stack((np.arange(0, 4), np.full(4, idx_y_x[1])))), board_state_pieces_colors_squares)
                            else:
                                board_state_pieces_colors_squares[0:2, 7, idx_y_x[1]] = o_img.promotion_array_piece_idxs[turn_i_current_opponent[0], selected_promotion_idx], turn_i_current_opponent[0]
                                o_img.draw_board((np.column_stack((np.arange(4, 8), np.full(4, idx_y_x[1])))), board_state_pieces_colors_squares)
                        o_img.promotion_ui_bbox_yx = None

                    elif o_img.piece_idx_selected is not None:
                        draw_selected_move(idx_y_x, o_img)
                    else:
                        draw_potential_moves(idx_y_x)

            elif o_img.click_is_on('menu', x, y):
                menu_button_bboxs = o_img.menus[o_img.current_menu]['bboxs']
                menu_button_idx = np.argwhere(np.logical_and(np.logical_and(menu_button_bboxs[:, 0] <= y, menu_button_bboxs[:, 1] >= y), np.logical_and(menu_button_bboxs[:, 2] <= x, menu_button_bboxs[:, 3] >= x))).flatten()
                o_img.mouse_xy = x, y
                if menu_button_idx.shape[0] != 0:
                    o_img.menus[o_img.current_menu]['funcs'][int(menu_button_idx)]()

            elif o_img.click_is_on('button', x, y):
                button_bboxs = o_img.button_ranges
                button_idx = np.argwhere(np.logical_and(np.logical_and(button_bboxs[:, 0] <= y, button_bboxs[:, 1] >= y), np.logical_and(button_bboxs[:, 2] <= x, button_bboxs[:, 3] >= x))).flatten()
                o_img.mouse_xy = x, y
                if button_idx.shape[0] != 0:
                    o_img.buttons[o_img.buttons_names[int(button_idx)]]['func']()

    board_state_pieces_colors_squares = np.zeros((3, 8, 8), dtype=np.uint8)
    board_state_pieces_colors_squares[2] = return_white_squares_grid()
    movement_vectors, movement_magnitudes = return_non_pawn_movement_vectors_and_magnitudes()
    o_img = OutputImg()

    for key in ('up', 'down', 'left', 'right'):
        keyboard.on_press_key(key, o_img.notation_keyboard_handler)

    rook_king_rook_has_not_moved, en_passant_tracker = np.ones((2, 3), dtype=np.uint0), np.zeros((2, 8), dtype=np.uint0)
    king_in_check_valid_piece_idxs, king_in_check_valid_moves = [None, None], [[], []]

    king_positions_yx, fisher_rook_xs = np.array([[7, 4], [0, 4]], dtype=np.int8), None
    king_closest_obstructing_pieces_is = 8 * np.ones((2, 8, 2), dtype=np.uint8)
    obstructing_piece_identities = (np.array([Pieces.queen.value, Pieces.rook.value], dtype=np.uint8),
                                    np.array([Pieces.queen.value, Pieces.bishop.value], dtype=np.uint8),
                                    np.array([Pieces.knight.value], dtype=np.uint8))

    cv2.namedWindow(o_img.window_name), cv2.setMouseCallback('Chess', mouse_handler)

    while 1:
        cv2.imshow(o_img.window_name, o_img.img)
        if new_game:
            set_starting_board_state(fisher=o_img.fisher)
            new_game = False
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    turn_i_current_opponent = (white_i, black_i)
    new_game, game_has_not_ended = True, True
    main()
