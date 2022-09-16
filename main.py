# It's chess in a day - one day to get the most functional 2p chess program up!
# Lets go!
import numpy as np
from enum import Enum, auto
import cv2, dir, personal_utils
from playsound import playsound

capture_sound, move_sound = dir.piece_capture_sound_dir, dir.piece_moved_sound_dir


class Img:
    total_image_size_yx = (720, 1280)
    label_offsets = .1

    grid_square_size_yx = int((total_image_size_yx[0] * (1 - label_offsets)) / 8), int((total_image_size_yx[0] * (1 - label_offsets)) / 8)
    x_label_offset = int(label_offsets * total_image_size_yx[0])
    icon_size_yx = (16, 16)

    valid_move_color = np.array([0, 0, 255], dtype=np.uint8)
    hover_color = np.array([0, 255, 255], dtype=np.uint8)
    ui_color = np.array([60, 60, 60], dtype=np.uint8)
    window_name = 'Chess'

    def __init__(self):
        self.board_img_bbox = np.array([0, self.grid_square_size_yx[0] * 8, self.x_label_offset, self.x_label_offset + self.grid_square_size_yx[1] * 8])
        self.img = np.zeros((self.total_image_size_yx[0], self.total_image_size_yx[1], 3), dtype=np.uint8)
        grid_square_templates = np.ones((2, 2, self.grid_square_size_yx[0], self.grid_square_size_yx[1], 3), dtype=np.uint8)
        grid_square_templates[0] = grid_square_templates[0] * (255, 255, 244)
        grid_square_templates[1] = grid_square_templates[1] * (180, 130, 170)
        for square in grid_square_templates:
            square[1] = square[0] * .35 + self.valid_move_color * .65
        self.grid_square_templates = grid_square_templates
        self.board_pieces_img_arrays, self.icon_capture_idx_ranges, self.icon_capture_relative_idxs, self.icon_capture_draw_points = self.return_img_arrays()
        self.promotion_array_piece_idxs = np.array([[Pieces.queen.value, Pieces.rook.value, Pieces.bishop.value, Pieces.knight.value],
                                                    [Pieces.knight.value, Pieces.bishop.value, Pieces.rook.value, Pieces.queen.value]], dtype=np.uint8) + 1
        self.piece_idx_selected, self.drawn_moves = None, None
        self.obstruction_next_vector_i, self.obstruction_next_piece_yx = None, None
        self.promotion_ui_bbox_yx = None

    def set_starting_board_state(self, position_array, fisher=False):
        if not fisher:
            non_pawn_row_values = np.array([Pieces.rook.value, Pieces.knight.value, Pieces.bishop.value, Pieces.queen.value, Pieces.king.value, Pieces.bishop.value, Pieces.knight.value, Pieces.rook.value]) + 1
        else:
            pass

        for piece_color_i, pawn_row_i, non_pawn_row_i, in zip((white_i, black_i), (6, 1), (7, 0)):
            for row_i, row_values in zip((pawn_row_i, non_pawn_row_i), (Pieces.pawn.value + 1, non_pawn_row_values[0:])):
                y_board_idxs, x_board_idxs = np.full(8, row_i), np.arange(0, 8)
                position_array[0, y_board_idxs, x_board_idxs] = row_values
                position_array[1, y_board_idxs, x_board_idxs] = piece_color_i
                self.draw_board((y_board_idxs, x_board_idxs), position_array)

        y_empty_square_idxs = np.indices((8, 8))
        self.draw_board((y_empty_square_idxs[0, 2:6].flatten(), y_empty_square_idxs[1, 2:6].flatten()), position_array)

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
                self.img[draw_idxs[draw_i, 0]: draw_idxs[draw_i, 1], draw_idxs[draw_i, 2]: draw_idxs[draw_i, 3]] = self.board_pieces_img_arrays[img_i[0] - 1, img_i[1], img_i[2], img_i[3]]

        if empty_spaces.shape[0] != 0:
            space_draw_is = position_array[2, position_array_draw_idxs_y_x[:, 0][empty_spaces], position_array_draw_idxs_y_x[:, 1][empty_spaces]]
            for draw_i, img_i in zip(empty_spaces, space_draw_is):
                self.img[draw_idxs[draw_i, 0]: draw_idxs[draw_i, 1], draw_idxs[draw_i, 2]: draw_idxs[draw_i, 3]] = self.grid_square_templates[img_i, final_color_i]

            print('b')
            pass

    def return_img_arrays(self):
        img_order = ('king', 'queen', 'bishop', 'knight', 'rook', 'pawn')
        img = cv2.imread(dir.chess_pieces_dir, cv2.IMREAD_UNCHANGED)
        img_white = img[0:img.shape[0] // 2]
        img_black = img[img.shape[0] // 2:]
        buffer_percentage = 1 - .25
        draw_start_y_x = (self.grid_square_size_yx[0] - (self.grid_square_size_yx[0] * buffer_percentage)) // 2, (self.grid_square_size_yx[1] - (self.grid_square_size_yx[1] * buffer_percentage)) // 2
        img_resize_y_x = (int(self.grid_square_size_yx[1] * buffer_percentage), int(self.grid_square_size_yx[0] * buffer_percentage))

        # Piece_type, Piece_color, square_color, capture_bool (1 = capturable, colored,
        img_arrays_pieces = np.zeros((len(img_order), 2, 2, 2, self.grid_square_size_yx[0], self.grid_square_size_yx[1], 3), dtype=np.uint8)
        icons_idx_ranges = np.zeros((2, len(img_order), 2), dtype=np.uint32)
        icon_relative_idxs = None
        icon_draw_points = None
        icon_idx_count = 0

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

                piece_capture_icon_img, piece_capture_icon_mask = cv2.resize(piece_native_size_img, (self.icon_size_yx[1], self.icon_size_yx[0]), interpolation=cv2.INTER_AREA), \
                                                                  cv2.resize(piece_native_size_mask, (self.icon_size_yx[1], self.icon_size_yx[0]), interpolation=cv2.INTER_AREA)

                piece_grid_mask_non_zero = np.column_stack(np.nonzero(piece_grid_mask))

                for square_color_i in (white_i, black_i):
                    piece_mask_non_zero_relative = piece_grid_mask_non_zero - piece_grid_mask.shape

                    for capture_state in (non_capture_draw_i, capture_draw_i):
                        piece_template = self.grid_square_templates[square_color_i, capture_state].copy()
                        piece_template[piece_mask_non_zero_relative[:, 0] - int(draw_start_y_x[0]), piece_mask_non_zero_relative[:, 1] - int(draw_start_y_x[1])] = piece_grid[piece_grid_mask_non_zero[:, 0], piece_grid_mask_non_zero[:, 1]][:, 0:3]
                        img_arrays_pieces[piece_i, piece_color_i, square_color_i, capture_state] = piece_template

                icon_nonzero = np.column_stack(np.nonzero(piece_capture_icon_mask))
                if icon_relative_idxs is None:
                    icon_relative_idxs = icon_nonzero - piece_capture_icon_mask.shape
                    icon_draw_points = piece_capture_icon_img[icon_nonzero]
                else:
                    icon_relative_idxs = np.vstack((icon_relative_idxs, (icon_nonzero - piece_capture_icon_mask.shape)))
                    icon_draw_points = np.vstack((icon_draw_points, piece_capture_icon_img[icon_nonzero]))
                icons_idx_ranges[piece_color_i, piece_icon_idx_count] = icon_idx_count, icon_idx_count + icon_nonzero.shape[0]
                icon_idx_count += icon_nonzero.shape[0]

        return img_arrays_pieces, icons_idx_ranges, icon_relative_idxs, icon_draw_points

    def click_is_on_board(self, mouse_x, mouse_y):
        if self.board_img_bbox[2] < mouse_x < self.board_img_bbox[3] and self.board_img_bbox[0] < mouse_y < self.board_img_bbox[1]:
            return True
        else:
            return False

    def return_draw_indicies_from_board_indicies(self, board_indicies_y_x):
        return np.column_stack((board_indicies_y_x[:, 0] * Img.grid_square_size_yx[0],
                                (board_indicies_y_x[:, 0] + 1) * Img.grid_square_size_yx[0],
                                board_indicies_y_x[:, 1] * Img.grid_square_size_yx[1] + self.x_label_offset,
                                (board_indicies_y_x[:, 1] + 1) * Img.grid_square_size_yx[1] + self.x_label_offset))

    def return_y_x_idx(self, mouse_x, mouse_y):
        return int((mouse_y / self.grid_square_size_yx[1])), \
               int((mouse_x - self.x_label_offset) / self.grid_square_size_yx[1])

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


def setup():
    chess_board_state_pieces_pieces_colors_square_colors = np.zeros((3, 8, 8), dtype=np.uint8)
    chess_board_state_pieces_pieces_colors_square_colors[2] = return_white_squares_grid()
    movement_vectors, movement_magnitudes = return_non_pawn_movement_vectors_and_magnitudes()

    return chess_board_state_pieces_pieces_colors_square_colors, movement_vectors, movement_magnitudes


def main():
    board_state_pieces_colors_squares, movement_vectors, movement_magnitudes = setup()

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
                    if magnitude_i == 1 and i == 1:
                        accepted_piece_ids = np.hstack((obstructing_piece_identities[i], Pieces.value.pawn + 1))
                    else:
                        accepted_piece_ids = obstructing_piece_identities[i]

                    if piece_id in accepted_piece_ids:
                        return True
                    else:
                        print('b')

        return False

    def return_potential_moves(piece_y_x_idx: tuple, movement_vectors_loc: tuple, magnitudes: tuple, check_type: str = 'moves', pawn_movement=False, pawn_attack=False, en_passant_check=None, obstruction_check=False) -> np.ndarray or None or False:
        """

        :type pawn_attack_en_passant: object
        """
        valid_squares_yx = []
        print('b')
        if obstruction_check:
            if np.any(np.all(king_closest_obstructing_pieces_is[current_turn_i] == piece_y_x_idx, axis=1)):
                obstruction_vector_i = np.argwhere((np.all(king_closest_obstructing_pieces_is[current_turn_i] == piece_y_x_idx, axis=1))).flatten()[0]
                next_piece_yx = return_potential_moves(piece_y_x_idx, (np.array([movement_vectors[Pieces.king.value][obstruction_vector_i]]),), (9,), check_type='obstruction')
                if next_piece_yx is not None:
                    next_piece_properties = board_state_pieces_colors_squares[0:2, next_piece_yx[:, 0], next_piece_yx[:, 1]]
                    potential_protection_vector = np.array([movement_vectors[Pieces.king.value][obstruction_vector_i]])
                    # If the next piece is a valid piece protecting it, the chosen piece can only move on that given vector
                    if next_piece_properties[1] != current_turn_i:
                        magnitude_i = np.max(np.abs(next_piece_yx - king_positions_yx[current_turn_i]))
                        if square_is_protected(potential_protection_vector, (next_piece_properties[0]), magnitude_i):
                            if np.any(np.all(movement_vectors_loc[0] == potential_protection_vector, axis=1)):
                                movement_vectors_loc = np.array([potential_protection_vector])
                            else:
                                return None

                    # Otherwise, the next piece is a piece of the same side, and set that as the potential next obstructing piece
                    else:
                        o_img.obstruction_next_vector_i, o_img.obstruction_next_piece_yx = potential_protection_vector, next_piece_yx
                        king_closest_obstructing_pieces_is[current_turn_i, obstruction_vector_i] = next_piece_yx[0]
                        print('obstruction changed')
                else:
                    king_closest_obstructing_pieces_is[current_turn_i, obstruction_vector_i] = (8, 8)

        for movement_vector, magnitude in zip(movement_vectors_loc, magnitudes):
            valid_vectors = np.ones(movement_vector.shape[0], dtype=np.uint0)
            for magnitude_i in range(1, magnitude):
                valid_vector_idxs = np.argwhere(valid_vectors).flatten()

                potential_moves = piece_y_x_idx + (movement_vector[valid_vector_idxs]) * magnitude_i
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
                                if current_turn_i == white_i:
                                    if en_passant_tracker[black_i, valid_potential_moves[empty_square_i, 1]]:
                                        valid_squares_yx.append(valid_potential_moves[empty_square_i])
                                else:
                                    if en_passant_tracker[white_i, valid_potential_moves[empty_square_i, 1]]:
                                        valid_squares_yx.append(valid_potential_moves[empty_square_i])

                    if valid_pieces.shape[0] != 0:
                        if not pawn_movement:
                            valid_opponent_pieces = np.argwhere(piece_identities_colors[1, valid_pieces] != current_turn_i).flatten()
                            if valid_opponent_pieces.shape[0] != 0:
                                valid_squares_yx.append(valid_potential_moves[valid_pieces[valid_opponent_pieces]])
                        valid_vectors[valid_vector_idxs[valid_pieces]] = 0
                else:
                    if valid_pieces.shape[0] != 0:
                        if check_type == 'obstruction':
                            valid_squares_yx.append(potential_moves[valid_square_is[valid_pieces]])
                        elif check_type == 'protection':
                            if valid_pieces.shape[0] != 0:
                                valid_ally_pieces = np.argwhere(piece_identities_colors[1, valid_pieces] != current_turn_i).flatten()
                                if valid_ally_pieces.shape[0] != 0:
                                    potential_protection_moves = valid_potential_moves[valid_pieces[valid_ally_pieces]]
                                    # Vector types are Orthogonal, Diagonal, and Knight

                                    if len(potential_protection_moves.shape) == 1:
                                        potential_protection_moves = np.array([[potential_protection_moves]])

                                    potential_protection_piece_ids = board_state_pieces_colors_squares[0, potential_protection_moves[:, 0], potential_protection_moves[:, 1]]
                                    potential_protection_vectors = movement_vector[valid_vector_idxs[valid_pieces[valid_ally_pieces]]]
                                    if square_is_protected(potential_protection_vectors, potential_protection_piece_ids, magnitude_i):
                                        return False

                        valid_vectors[valid_vector_idxs[valid_pieces]] = 0

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

    def set_obstructing_piece_idxs(vectors, initial_setup=False):
        nonlocal king_closest_obstructing_pieces_is
        if initial_setup:
            for i, king_position_yx in enumerate(king_positions_yx):
                moves = king_position_yx + vectors
                valid_square_is = np.argwhere(np.all(np.logical_and(moves >= 0, moves <= 7), axis=1)).flatten()
                king_closest_obstructing_pieces_is[i, valid_square_is] = moves[valid_square_is]

    def set_starting_board_state(fisher=False):
        nonlocal rook_king_rook_has_not_moved, en_passant_tracker, king_closest_obstructing_pieces_is
        if not fisher:
            non_pawn_row_values = np.array([Pieces.rook.value, Pieces.knight.value, Pieces.bishop.value, Pieces.queen.value, Pieces.king.value, Pieces.bishop.value, Pieces.knight.value, Pieces.rook.value]) + 1
        else:
            pass

        for piece_color_i, pawn_row_i, non_pawn_row_i, in zip((white_i, black_i), (6, 1), (7, 0)):
            for row_i, row_values in zip((pawn_row_i, non_pawn_row_i), (Pieces.pawn.value + 1, non_pawn_row_values[0:])):
                y_board_idxs, x_board_idxs = np.full(8, row_i), np.arange(0, 8)
                board_state_pieces_colors_squares[0, y_board_idxs, x_board_idxs] = row_values
                board_state_pieces_colors_squares[1, y_board_idxs, x_board_idxs] = piece_color_i
                o_img.draw_board(np.column_stack((y_board_idxs, x_board_idxs)), board_state_pieces_colors_squares)

        y_empty_square_idxs = np.indices((8, 8))
        o_img.draw_board((np.column_stack((y_empty_square_idxs[0, 2:6].flatten(), y_empty_square_idxs[1, 2:6].flatten()))), board_state_pieces_colors_squares)
        rook_king_rook_has_not_moved, en_passant_tracker = np.ones((2, 3), dtype=np.uint0), np.zeros((2, 8), dtype=np.uint0)
        set_obstructing_piece_idxs(movement_vectors[Pieces.king.value], initial_setup=True)

    def draw_potential_moves(piece_yx_idx):
        nonlocal board_state_pieces_colors_squares, rook_king_rook_has_not_moved
        if board_state_pieces_colors_squares[1, piece_yx_idx[0], piece_yx_idx[1]] == current_turn_i:
            piece_id = board_state_pieces_colors_squares[0, piece_yx_idx[0], piece_yx_idx[1]]
            valid_squares_yx = None
            if piece_id != 0:
                if piece_id < 5:
                    vectors, magnitude = movement_vectors[piece_id - 1], movement_magnitudes[piece_id - 1]
                    valid_squares_yx = return_potential_moves(piece_yx_idx, (vectors,), (magnitude,), obstruction_check=True)
                # King Handling
                elif piece_id == 5:
                    vectors, magnitude = movement_vectors[piece_id - 1], movement_magnitudes[piece_id - 1]
                    single_moves = return_potential_moves(piece_yx_idx, (vectors,), (magnitude,))
                    if single_moves is not None:
                        valid_single_moves = np.ones(single_moves.shape[0], dtype=np.uint0)
                        for i, single_move in enumerate(single_moves):
                            square_is_not_protected = return_potential_moves(single_move, (movement_vectors[Pieces.queen.value], movement_vectors[Pieces.knight.value],),
                                                                             (movement_magnitudes[Pieces.queen.value], movement_magnitudes[Pieces.knight.value]), check_type='protection')
                            if square_is_not_protected is not None:
                                valid_single_moves[i] = 0
                        # Castle Check
                        if np.any(valid_single_moves != 0):
                            valid_squares_yx = single_moves[np.argwhere(valid_single_moves).flatten()]
                            if rook_king_rook_has_not_moved[current_turn_i, 1]:
                                for rook_has_moved_i, rook_x_value, king_x_offset in zip((0, 2), (0, 7), (-2, 2)):
                                    if rook_king_rook_has_not_moved[current_turn_i, rook_has_moved_i]:
                                        castle_x_idxs = np.arange(min(rook_x_value, piece_yx_idx[1]), max(rook_x_value, piece_yx_idx[1]))
                                        castle_idxs_all = np.column_stack((np.full_like(castle_x_idxs, piece_yx_idx[0]), castle_x_idxs))

                                        if np.all(board_state_pieces_colors_squares[0, castle_idxs_all[:, 0], castle_idxs_all[:, 1]]) == 0:
                                            castle_bool = True
                                            for castle_idx in castle_idxs_all:
                                                square_is_not_protected = return_potential_moves(castle_idx, (movement_vectors[Pieces.queen.value], movement_vectors[Pieces.knight.value],),
                                                                                                 (movement_magnitudes[Pieces.queen.value], movement_magnitudes[Pieces.knight.value]), check_type='protection')
                                                if square_is_not_protected is not None:
                                                    castle_bool = False
                                                    break
                                            if castle_bool:
                                                valid_squares_yx = np.vstack((valid_squares_yx, np.array([piece_yx_idx[0], piece_yx_idx[1] + king_x_offset], dtype=np.uint8)))

                # Pawn Handling
                elif piece_id == 6:
                    movement_vector_pawn, movement_magnitude = movement_vectors[Pieces.pawn_movement.value], movement_magnitudes[Pieces.pawn_movement.value]
                    attack_vector_pawn, attack_magnitude = movement_vectors[Pieces.pawn_attack.value], movement_magnitudes[Pieces.pawn_attack.value]
                    en_passant_check = False
                    if current_turn_i == white_i:
                        if piece_yx_idx[0] == 3:
                            en_passant_check = True
                        elif piece_yx_idx[0] == 6:
                            movement_magnitude += 1
                    else:
                        movement_vector_pawn = movement_vector_pawn.copy() * -1
                        attack_vector_pawn = attack_vector_pawn.copy() * -1
                        if piece_yx_idx[0] == 4:
                            en_passant_check = True
                        elif piece_yx_idx[0] == 1:
                            movement_magnitude += 1

                    valid_squares_yx_movement = return_potential_moves(piece_yx_idx, (movement_vector_pawn,), (movement_magnitude,), check_type='moves', pawn_movement=True, obstruction_check=True)
                    valid_squares_yx_attack = return_potential_moves(piece_yx_idx, (attack_vector_pawn,), (attack_magnitude,), check_type='moves', pawn_attack=True, en_passant_check=en_passant_check, obstruction_check=True)
                    if valid_squares_yx_movement is not None:
                        if valid_squares_yx_attack is not None:
                            valid_squares_yx = np.vstack((valid_squares_yx_movement, valid_squares_yx_attack))
                        else:
                            valid_squares_yx = valid_squares_yx_movement
                    elif valid_squares_yx_attack is not None:
                        valid_squares_yx = valid_squares_yx_attack

            # Draws Potential moves if available
            if valid_squares_yx is not None:
                o_img.draw_board(valid_squares_yx, board_state_pieces_colors_squares, pre_move=True)
                o_img.piece_idx_selected = piece_yx_idx
                o_img.drawn_moves = valid_squares_yx

    def draw_selected_move(piece_selected_yx, output_img_loc: Img):
        nonlocal current_turn_i
        piece_initial_yx, drawn_idxs = output_img_loc.piece_idx_selected, output_img_loc.drawn_moves
        drawn_idx_compare = np.all(drawn_idxs == piece_selected_yx, axis=1)
        in_drawn_idxs = np.argwhere(drawn_idx_compare).flatten()
        if in_drawn_idxs.shape[0] != 0:
            draw_idxs = np.vstack((piece_initial_yx, piece_selected_yx))
            current_piece_i = board_state_pieces_colors_squares[0, piece_initial_yx[0], piece_initial_yx[1]] - 1

            if current_piece_i == Pieces.rook.value:
                if piece_initial_yx[1] == 7:
                    rook_king_rook_has_not_moved[current_turn_i, 2] = 0
                    print('b')
                elif piece_initial_yx[1] == 0:
                    rook_king_rook_has_not_moved[current_turn_i, 0] = 0
                    print('b')
            elif current_piece_i == Pieces.king.value:
                if piece_initial_yx[1] == 4:
                    # Castle
                    if abs(piece_selected_yx[1] - piece_initial_yx[1]) == 2:
                        if piece_selected_yx[1] > piece_initial_yx[1]:
                            board_state_pieces_colors_squares[0:2, piece_initial_yx[0], piece_selected_yx[1] - 1] = Pieces.rook.value + 1, current_turn_i
                            board_state_pieces_colors_squares[0:2, piece_initial_yx[0], 7] = 0, 0

                            draw_idxs = np.vstack((draw_idxs, np.array([[piece_initial_yx[0], piece_selected_yx[1] - 1],
                                                                        [piece_initial_yx[0], 7]])))

                        else:
                            board_state_pieces_colors_squares[0:2, piece_initial_yx[0], piece_selected_yx[1] + 1] = Pieces.rook.value + 1, current_turn_i
                            board_state_pieces_colors_squares[0:2, piece_initial_yx[0], 0] = 0
                            draw_idxs = np.vstack((draw_idxs, np.array([[piece_initial_yx[0], piece_selected_yx[1] + 1],
                                                                        [piece_initial_yx[0], 0]])))
                rook_king_rook_has_not_moved[current_turn_i, 1] = 0
                king_positions_yx[current_turn_i] = piece_initial_yx

            elif current_piece_i == Pieces.pawn.value:
                if abs(piece_initial_yx[0] - piece_selected_yx[0]) == 2:
                    en_passant_tracker[current_turn_i, piece_initial_yx[1]] = 1
                # En Passant Capture, no piece present, set the board state capture
                else:
                    # Capture made, check for en passant draw necessity
                    if piece_selected_yx[1] - piece_initial_yx[1] != 0:
                        # Black Capture
                        if piece_selected_yx[0] - piece_initial_yx[0] == 1:
                            # Black Capture En Passant
                            if board_state_pieces_colors_squares[0, piece_selected_yx[0], piece_selected_yx[1]] == 0:
                                board_state_pieces_colors_squares[0:2, piece_selected_yx[0] - 1, piece_selected_yx[1]] = (0, 0)
                                draw_idxs = np.vstack((draw_idxs, (piece_selected_yx[0] - 1, piece_selected_yx[1])))
                        # White Capture
                        else:
                            if board_state_pieces_colors_squares[0, piece_selected_yx[0], piece_selected_yx[1]] == 0:
                                board_state_pieces_colors_squares[0:2, piece_selected_yx[0] + 1, piece_selected_yx[1]] = (0, 0)
                                draw_idxs = np.vstack((draw_idxs, (piece_selected_yx[0] + 1, piece_selected_yx[1])))
                            print('b')

            obstruction_vector = piece_selected_yx - king_positions_yx[current_turn_i]
            # Obstruction check, if the selected move is on the same axis as the king, check to see if the piece is the closest obstructing piece
            # Orthogonal check, x_y axis
            test = np.sum(np.abs(obstruction_vector)) % 2 == 0
            test_2 = np.abs(obstruction_vector)[0] == np.abs(obstruction_vector)[1]

            if np.any(obstruction_vector == 0):
                obstruction_idx = np.argwhere(obstruction_vector != 0).flatten()[0]
                obstruction_vector_sign = np.sign(obstruction_vector)
                obstruction_tracker_i = np.argwhere(np.all(movement_vectors[Pieces.king.value] == obstruction_vector_sign, axis=1))
                print('b')
                # Magnitude is not 1
                if np.abs(obstruction_vector[obstruction_idx] != 1):
                    if piece_selected_yx[obstruction_idx] > king_positions_yx[current_turn_i, obstruction_idx]:
                        variable_obstruction_idxs = np.arange(king_positions_yx[current_turn_i, obstruction_idx], piece_selected_yx[obstruction_idx])
                    else:
                        variable_obstruction_idxs = np.arange(piece_selected_yx[obstruction_idx], king_positions_yx[current_turn_i, obstruction_idx])

                    if obstruction_idx == 0:
                        compare_idxs_yx = np.column_stack((variable_obstruction_idxs, np.full_like(variable_obstruction_idxs, king_positions_yx[current_turn_i, 1])))
                    else:
                        compare_idxs_yx = np.column_stack((np.full_like(variable_obstruction_idxs, king_positions_yx[current_turn_i, 0]), variable_obstruction_idxs))
                    #Checks to see if the selected pieces is in compare_idxs_yx, if it is, remove it

                    compare_idxs_yx = compare_idxs_yx[np.invert(np.all(compare_idxs_yx == piece_initial_yx, axis=1))]

                    pieces = board_state_pieces_colors_squares[0, compare_idxs_yx[:, 0], compare_idxs_yx[:, 1]]
                    if np.all(pieces == 0):
                        king_closest_obstructing_pieces_is[current_turn_i, obstruction_tracker_i] = piece_selected_yx
                # Magnitude is 1, piece is adjacent to king
                else:
                    king_closest_obstructing_pieces_is[current_turn_i, obstruction_tracker_i] = piece_selected_yx
            # a = 0
            #Diagonal check

            elif np.sum(np.abs(obstruction_vector)) % 2 == 0 and np.abs(obstruction_vector)[0] == np.abs(obstruction_vector)[1]:
                obstruction_vector_sign = np.sign(obstruction_vector)
                obstruction_tracker_i = np.argwhere(np.all(movement_vectors[Pieces.king.value] == obstruction_vector_sign, axis=1))
                # Magnitude is not 1
                if np.abs(obstruction_vector[0]) != 1:
                    if piece_selected_yx[0] > king_positions_yx[current_turn_i, 0]:
                        start_y_idx, end_y_idx = king_positions_yx[current_turn_i, 0], piece_selected_yx[0]
                    else:
                        start_y_idx, end_y_idx = piece_selected_yx[0], king_positions_yx[current_turn_i, 0]

                    if piece_selected_yx[1] > king_positions_yx[current_turn_i, 1]:
                        start_x_idx, end_x_idx = king_positions_yx[current_turn_i, 0], piece_selected_yx[current_turn_i, 0]
                    else:
                        start_x_idx, end_x_idx = piece_selected_yx[0], king_positions_yx[current_turn_i, 0]

                    compare_idxs_yx = board_state_pieces_colors_squares[0, np.arange(start_y_idx, end_y_idx), np.arange(start_x_idx, end_x_idx)]
                    compare_idxs_yx = compare_idxs_yx[np.invert(np.all(compare_idxs_yx == piece_initial_yx, axis=1))]

                    if np.all(compare_idxs_yx == 0):
                        king_closest_obstructing_pieces_is[current_turn_i, obstruction_tracker_i] = piece_selected_yx
                else:
                    king_closest_obstructing_pieces_is[current_turn_i, obstruction_tracker_i] = piece_selected_yx


            board_state_pieces_colors_squares[0:2, piece_selected_yx[0], piece_selected_yx[1]] = board_state_pieces_colors_squares[0:2, piece_initial_yx[0], piece_initial_yx[1]]
            board_state_pieces_colors_squares[0:2, piece_initial_yx[0], piece_initial_yx[1]] = (0, 0)
            output_img_loc.draw_board(draw_idxs, board_state_pieces_colors_squares)
            output_img_loc.draw_board(drawn_idxs, board_state_pieces_colors_squares)
            output_img_loc.piece_idx_selected, output_img_loc.drawn_moves = None, None

            # Promotion Handler
            if current_piece_i == Pieces.pawn.value:
                if (current_turn_i == black_i and piece_selected_yx[0] == 7) or (current_turn_i == white_i and piece_selected_yx[0] == 0):
                    o_img.draw_promotion_selector(piece_selected_yx[1], board_state_pieces_colors_squares, current_turn_i)

                    while o_img.promotion_ui_bbox_yx is not None:
                        cv2.imshow(o_img.window_name, o_img.img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            break

            if current_turn_i == white_i:
                current_turn_i = black_i
            else:
                current_turn_i = white_i
            en_passant_tracker[current_turn_i] = 0
            cv2.imshow(o_img.window_name, o_img.img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
            '''if sound_state != 0:
                playsound(capture_sound)
            else:
                playsound(move_sound)'''




        elif np.logical_and(board_state_pieces_colors_squares[0, piece_selected_yx[0], piece_selected_yx[1]] != 0,
                            board_state_pieces_colors_squares[1, piece_selected_yx[0], piece_selected_yx[1]] == current_turn_i):
            output_img_loc.draw_board(drawn_idxs, board_state_pieces_colors_squares)
            draw_potential_moves(piece_selected_yx)

    o_img = Img()
    current_turn_i = white_i

    move_count, minimum_theoretical_stalemate_move_count = 0, 10
    rook_king_rook_has_not_moved, en_passant_tracker, king_in_check, king_checking_piece_idx = np.zeros((2, 3), dtype=np.uint0), np.zeros((2, 8), dtype=np.uint0), np.array([0, 0], dtype=np.uint0), np.zeros((2, 2), dtype=np.uint8)
    king_positions_yx = np.array([[7, 4], [0, 4]], dtype=np.uint8)
    king_closest_obstructing_pieces_is = 8 * np.ones((2, 9, 2), dtype=np.uint8)
    obstructing_piece_identities = (np.array([Pieces.queen.value, Pieces.rook.value, Pieces.king.value], dtype=np.uint8) + 1,
                                    np.array([Pieces.queen.value, Pieces.bishop.value, Pieces.king.value], dtype=np.uint8) + 1,
                                    np.array([Pieces.knight.value], dtype=np.uint8) + 1)
    set_starting_board_state()
    cv2.namedWindow(Img.window_name)

    def mouse_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if o_img.click_is_on_board(x, y):
                idx_y_x = o_img.return_y_x_idx(x, y)
                if o_img.promotion_ui_bbox_yx is not None:
                    if o_img.promotion_ui_bbox_yx[0] < y < o_img.promotion_ui_bbox_yx[1] and o_img.promotion_ui_bbox_yx[2] < x < o_img.promotion_ui_bbox_yx[3]:
                        selected_promotion_idx = ((y - o_img.promotion_ui_bbox_yx[0]) // o_img.grid_square_size_yx[0])
                        if current_turn_i == white_i:
                            board_state_pieces_colors_squares[0:2, 0, idx_y_x[1]] = o_img.promotion_array_piece_idxs[current_turn_i, selected_promotion_idx], current_turn_i
                            o_img.draw_board((np.column_stack((np.arange(0, 4), np.full(4, idx_y_x[1])))), board_state_pieces_colors_squares)
                        else:
                            board_state_pieces_colors_squares[0:2, 7, idx_y_x[1]] = o_img.promotion_array_piece_idxs[current_turn_i, selected_promotion_idx], current_turn_i
                            o_img.draw_board((np.column_stack((np.arange(4, 8), np.full(4, idx_y_x[1])))), board_state_pieces_colors_squares)
                    o_img.promotion_ui_bbox_yx = None

                elif o_img.piece_idx_selected is not None:
                    draw_selected_move(idx_y_x, o_img)
                    pass
                else:
                    draw_potential_moves(idx_y_x)

                print('on_board')

    cv2.setMouseCallback('Chess', mouse_handler)

    while 1:
        cv2.imshow(o_img.window_name, o_img.img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
