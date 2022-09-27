import os.path

path = os.path.dirname(os.path.realpath(__file__))

img_dir = f'{path}/imgs'
chess_pieces_img = f'{img_dir}/chess_pieces.png'
settings_cog_img =f'{img_dir}/settings_cog.png'

sound_dir = f'{path}/sounds'
start_sound = f'{sound_dir}/war_horn.wav'
move_sound = f'{sound_dir}/piece_moved.wav'
victory_sound = f'{sound_dir}/victory.wav'
capture_sound = f'{sound_dir}/piece_capture.wav'

export_directory = f'{path}/notation_exports/'

