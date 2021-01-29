import os
from data.pix2pix_dataset import Pix2pixDataset


class Day2NightDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(aspect_ratio=2.0)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def get_paths(self, opt):
        croot = opt.croot
        sroot = opt.sroot

        with open(os.path.join(croot, 'bdd100k_lists/day2night/day_%s.txt' % opt.phase)) as c_list:
            c_image_paths_read = c_list.read().splitlines()
            c_image_paths = [os.path.join(croot, p) for p in c_image_paths_read if p != '']

        with open(os.path.join(sroot, 'bdd100k_lists/day2night/night_%s.txt' % opt.phase)) as s_list:
            s_image_paths_read = s_list.read().splitlines()
            s_image_paths = [os.path.join(sroot, p) for p in s_image_paths_read if p != '']

        if opt.phase == 'train':
            c_image_paths = c_image_paths + c_image_paths

        instance_paths = []

        length = min(len(c_image_paths), len(s_image_paths))
        c_image_paths = c_image_paths[:length]
        s_image_paths = s_image_paths[:length]
        return c_image_paths, s_image_paths, instance_paths

    def paths_match(self, path1, path2):
        return True