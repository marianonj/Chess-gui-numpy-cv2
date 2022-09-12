# It's chess in a day - one day to get the most functional 2p chess program up!
# Lets go!
import numpy as np
from enum import Enum, auto
import cv2, dir, personal_utils


class Img:
    total_image_size_yx = (720, 1280)
    label_offsets = .05

    grid_square_size_yx = int((total_image_size_yx[0] * (1 - label_offsets)) / 8), int((total_image_size_yx[0] * (1 - label_offsets)) / 8)
    x_label_offset = int(label_offsets * total_image_size_yx[0])
    icon_size_yx = (16, 16)

    valid_move_color = np.array([0, 0, 255], dtype=np.uint8)
    hover_color = np.array([0, 255, 255], dtype=np.uint8)
    ui_color = np.array([60, 60, 60], dtype=np.uint8)

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
        self.piece_idx_selected, self.drawn_potential_moves = None, None

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
        draw_idxs = self.return_draw_indicies_from_board_indicies(position_array_draw_idxs_y_x[0], position_array_draw_idxs_y_x[1])
        pieces = np.argwhere(position_array[0, position_array_draw_idxs_y_x[0], position_array_draw_idxs_y_x[1]] != 0).flatten()
        empty_spaces = np.argwhere(position_array[0, position_array_draw_idxs_y_x[0], position_array_draw_idxs_y_x[1]] == 0).flatten()
        if pre_move:
            final_color_i = 1
        else:
            final_color_i = 0
        if pieces.shape[0] != 0:
            piece_draw_is = np.vstack((position_array[0:3, position_array_draw_idxs_y_x[0][pieces], position_array_draw_idxs_y_x[1][pieces]], np.full(pieces.shape[0], final_color_i)))
            for draw_i, img_i in zip(pieces, piece_draw_is.T):
                self.img[draw_idxs[draw_i, 0]: draw_idxs[draw_i, 1], draw_idxs[draw_i, 2]: draw_idxs[draw_i, 3]] = self.board_pieces_img_arrays[img_i[0] - 1, img_i[1], img_i[2], img_i[3]]

        if empty_spaces.shape[0] != 0:
            space_draw_is = position_array[2, position_array_draw_idxs_y_x[0][empty_spaces], position_array_draw_idxs_y_x[1][empty_spaces]]
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

    def return_draw_indicies_from_board_indicies(self, board_indicies_y, board_indicies_x):
        return np.column_stack((board_indicies_y * Img.grid_square_size_yx[0],
                                (board_indicies_y + 1) * Img.grid_square_size_yx[0],
                                board_indicies_x * Img.grid_square_size_yx[1] + self.x_label_offset,
                                (board_indicies_x + 1) * Img.grid_square_size_yx[1] + self.x_label_offset))

    def return_y_x_idx(self, mouse_x, mouse_y):
        return int((mouse_y / self.grid_square_size_yx[1])), \
               int((mouse_x - self.x_label_offset) / self.grid_square_size_yx[1])


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


def return_non_pawn_movement_vectors_and_magnitudes():
    # All vectors, per numpy convention , are y, x movements
    orthogonal_vectors = np.array([[-1, 0],
                                   [1, 0],
                                   [0, 1],
                                   [0, -1]], dtype=np.int8)
    diagonal_vectors = np.array([[-1, -1],
                                 [-1, 1],
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
    vector_magnitudes = np.array([2, 9, 9, 9, 2], dtype=np.uint8)

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
    def return_potential_moves(calculation_method):
        if type == 'p':
            pass
        elif type == 'k':
            pass
        else:

            pass

        pass

    def draw_potential_moves(idx_y_x):
        nonlocal chess_board_state_pieces_pieces_colors_square_colors
        if chess_board_state_pieces_pieces_colors_square_colors[1, idx_y_x[0], idx_y_x[1]] == current_turn_i:
            piece_id = chess_board_state_pieces_pieces_colors_square_colors[0, idx_y_x[0], idx_y_x[1]]
            valid_squares_y_x = []
            if piece_id != 0:
                if piece_id < 5:
                    vectors, magnitudes = movement_vectors[piece_id - 1], movement_magnitudes[piece_id - 1]
                    valid_vectors = np.ones(vectors.shape[0])

                    while np.any(valid_vectors) == 1:
                        print('b')
                        for i in range(1, magnitudes):
                            valid_vector_is = np.argwhere(valid_vectors).flatten()
                            potential_moves = idx_y_x + vectors
                            valid_squares = np.all(np.logical_and(potential_moves >= 0, potential_moves < 8), axis=1)
                            valid_square_is = np.argwhere(valid_squares).flatten()
                            valid_vectors[np.invert(valid_squares)] = 0

                            piece_identities_colors = chess_board_state_pieces_pieces_colors_square_colors[0:2, potential_moves[valid_square_is][:, 0], potential_moves[valid_square_is][:, 1]]
                            empty_squares = np.argwhere(piece_identities_colors[0] == 0).flatten()
                            valid_pieces = np.argwhere(np.logical_and(piece_identities_colors[0] != 0, piece_identities_colors[1] != current_turn_i)).flatten()
                            if empty_squares.shape[0] != 0:
                                for empty_square in empty_squares:
                                    valid_squares_y_x.append(potential_moves[valid_square_is[empty_square]])
                            if valid_pieces.shape[0] != 0:
                                for valid_piece in valid_pieces:
                                    valid_squares_y_x.append(potential_moves[valid_square_is[valid_piece]])


                            if len(valid_squares_y_x) != 0:
                                if len(valid_squares_y_x) == 1:
                                    moves_stacked = valid_squares_y_x[0]
                                    output_img.draw_board((moves_stacked[0], moves_stacked[1]), chess_board_state_pieces_pieces_colors_square_colors)
                                else:
                                    moves_stacked = np.vstack(valid_squares_y_x)
                                    output_img.draw_board((moves_stacked[:, 0], moves_stacked[:, 1]), chess_board_state_pieces_pieces_colors_square_colors, pre_move=True)
                                output_img.piece_idx_selected = idx_y_x
                                output_img.drawn_potential_moves = moves_stacked
                            if i == magnitudes - 1:
                                valid_vectors = np.array([0])
                # King Handling
                elif piece_id == 5:
                    pass
                # Pawn Handling
                elif piece_id == 6:
                    pass

    def draw_moves(idx_y_x, output_img_loc:Img):
        nonlocal current_turn_i
        selected_idx, drawn_idxs = output_img_loc.piece_idx_selected, output_img_loc.drawn_potential_moves
        drawn_idx_compare = np.all(drawn_idxs == idx_y_x, axis=1)
        in_drawn_idxs = np.argwhere(drawn_idx_compare).flatten()
        if in_drawn_idxs.shape[0] != 0:
            chess_board_state_pieces_pieces_colors_square_colors[0:2, idx_y_x[0], idx_y_x[1]] = chess_board_state_pieces_pieces_colors_square_colors[0:2, selected_idx[0], selected_idx[1]]
            chess_board_state_pieces_pieces_colors_square_colors[0:2, selected_idx[0], selected_idx[1]] = (0, 0)
            draw_idxs = np.vstack((selected_idx, idx_y_x))
            output_img_loc.draw_board((draw_idxs[:, 0], draw_idxs[:, 1]), chess_board_state_pieces_pieces_colors_square_colors)
            output_img_loc.draw_board((drawn_idxs[:, 0], drawn_idxs[:, 1]), chess_board_state_pieces_pieces_colors_square_colors)
            output_img_loc.piece_idx_selected, output_img_loc.drawn_potential_moves = None, None
            if current_turn_i == white_i:
                current_turn_i = black_i
            else:
                current_turn_i = white_i

    output_img = Img()
    current_turn_i = white_i

    chess_board_state_pieces_pieces_colors_square_colors, movement_vectors, movement_magnitudes = setup()
    en_passant_tracker = np.zeros((2, 8), dtype=np.uint0)
    output_img.set_starting_board_state(chess_board_state_pieces_pieces_colors_square_colors)
    cv2.namedWindow('Chess')
    def mouse_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if output_img.click_is_on_board(x, y):
                idx_y_x = output_img.return_y_x_idx(x, y)
                if output_img.piece_idx_selected is not None:
                    draw_moves(idx_y_x, output_img)
                    pass
                else:
                    draw_potential_moves(idx_y_x)


                print('on_board')

    cv2.setMouseCallback('Chess', mouse_handler)

    while 1:
        cv2.imshow('Chess', output_img.img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
