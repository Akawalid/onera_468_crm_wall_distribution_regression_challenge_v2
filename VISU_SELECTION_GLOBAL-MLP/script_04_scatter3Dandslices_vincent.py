#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

#####################################################################################

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Input YAML configuration file
config_filename = "./config.yaml"

# Dictionary to adjust camera position & margins
camera_settings = {
    'elevation': 50., # 20.,
    'azimuth': 120.,
    'zoom': 1.,
    'xlim': [None, None],
    'ylim': [None, 5.],
    'zlim': [None, None],
    'xoffsets': [10., -14.],
    'yoffsets': [0., 0.],
    'zoffsets': [0., 0.]
}

# Dictionary to store plotting options
plot_settings = {
    'figsize': None,
    'dpi': 400,
    'output_folder': '.',
    'filename': 'figscatter',
    'title': None,
    'title_fontsize': 12,
    'cmap': 'jet',  # others can be: 'hsv', 'RdBu' 'PiYG'
    'cbar': {
        'title': 'cf',
        'title_fontsize': 8,
        'ticks_fontsize': 6
    }
}

###################################################################################################

def set_logger():
    class ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG:    "\033[36m",  # cyan
            logging.INFO:     "\033[32m",  # green
            logging.WARNING:  "\033[33m",  # yellow
            logging.ERROR:    "\033[31m",  # red
            logging.CRITICAL: "\033[41m",  # red background
        }
        RESET = "\033[0m"

        def format(self, record):
            color = self.COLORS.get(record.levelno, self.RESET)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
            return super().format(record)
        
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = ColorFormatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

#==============================================================================        

def plot_scatter3D(x, y, z, **kwargs):
    """
    Draw a 3D scatter plot for multiple variables defined in a config dictionary
    """
    # Shorthands
    camera_settings = kwargs["camera_settings"]
    plot_settings = kwargs["plot_settings"]

    config = kwargs["config"]
    minf = config['Minf']
    aoa = config['AoA']
    pi = config['Pi']
    nplots = len(config["variable_names"])
    vmins = [r[0] for r in config["ranges"]]
    vmaxs = [r[1] for r in config["ranges"]]
    
    # Sanitise inputs
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)
    if camera_settings['xlim'][0] is None:
        camera_settings['xlim'][0] = xmin
    if camera_settings['xlim'][1] is None:
        camera_settings['xlim'][1] = xmax
    if camera_settings['ylim'][0] is None:
        camera_settings['ylim'][0] = ymin
    if camera_settings['ylim'][1] is None:
        camera_settings['ylim'][1] = ymax
    if camera_settings['zlim'][0] is None:
        camera_settings['zlim'][0] = zmin
    if camera_settings['zlim'][1] is None:
        camera_settings['zlim'][1] = zmax

    ax_nrowmax = 1
    ax_ncolmax = None
    if plot_settings['right_panel']['show']:
        n2dplots = len(plot_settings['right_panel']['figure_rootnames'])
        gs = gridspec.GridSpec(
            nrows=n2dplots+1, ncols=4,
            width_ratios=[1.]*n2dplots + [1.],
            height_ratios=[20./n2dplots]*n2dplots + [1.]
        )
        ax_nrowmax = n2dplots
        ax_ncolmax = -1
    else:
        gs = gridspec.GridSpec(
            nrows=2, ncols=3,
            width_ratios=[1., 1., 1.],
            height_ratios=[20., 1.]
        )

    # Loop over variables to plot
    for i in range(nplots):
        varname = config["variable_names"][i]
        logger.info(f"    + Variable {varname}")

        # Load data
        var = np.load(config["filenames"][i])

        # Create figure
        if plot_settings['figsize'] is None:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=plot_settings['figsize'])

        # 3D scatter
        ax = fig.add_subplot(
            gs[:ax_nrowmax,:ax_ncolmax],
            projection=Axes3D.name
        )
        sca = ax.scatter3D(
            x, y, z, vmin=vmins[i], vmax=vmaxs[i], c=var,
            cmap=plt.get_cmap(plot_settings['cmap']),
            clip_on=False
        )
        
        ax.view_init(
            elev=camera_settings['elevation'],
            azim=camera_settings['azimuth']
        )
        ax.set(
            xlim=(
                camera_settings['xlim'][0] + camera_settings['xoffsets'][0],
                camera_settings['xlim'][1] + camera_settings['xoffsets'][1]
            ),
            ylim=(
                camera_settings['ylim'][0] + camera_settings['yoffsets'][0],
                camera_settings['ylim'][1] + camera_settings['yoffsets'][1]
            ),
            zlim=(
                camera_settings['zlim'][0] + camera_settings['zoffsets'][0],
                camera_settings['zlim'][1] + camera_settings['zoffsets'][1]
            )
        )
        
        ax.set_axis_off()
        limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        ax.set_box_aspect(np.ptp(limits, axis=1), zoom=camera_settings['zoom'])
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Colorbar
        cax = fig.add_subplot(gs[-1,1])
        cbar = fig.colorbar(sca, cax=cax, orientation='horizontal') #shrink=0.5) 
        cbar.set_label(
            varname,
            size=plot_settings['cbar']['title_fontsize']
        )
        cbar.ax.tick_params(labelsize=plot_settings['cbar']['ticks_fontsize'])
        cbar.ax.xaxis.set_label_position('top')

        # Add 2D graphs
        filename_suffix = f"_Minf_{minf}_AoA_{aoa}_Pi_{pi}"
        root_varname = varname.replace('hat', '')
        if plot_settings['right_panel']['show']:
            figpaths = plot_settings['right_panel']['figure_rootnames']
            for j, figpath in enumerate(figpaths):
                ax2d = fig.add_subplot(gs[j, -1])
                figpath = figpath.rsplit('_', 1)
                figpath = f"{figpath[0]}{filename_suffix}_{figpath[1]}_{root_varname}{root_varname}hat.png"
                img = mpimg.imread(figpath)
                ax2d.imshow(img)
                ax2d.axis("off")

        title = plot_settings.get('title', None)
        if title is not None:
            long_varname = config["long_variable_names"][i]
            title = f"{long_varname} - {title}, Minf = {minf}, AoA = {aoa}, Pi={pi} 10^6"
            fig.suptitle(title, fontsize=plot_settings['title_fontsize'])
        fig.tight_layout(pad=0.)
        output_folder = plot_settings['output_folder']
        filename = plot_settings['filename']
        if filename.endswith(".png"):
            filename = filename[:-4]
        filename += f"{filename_suffix}_{varname}"
        filename = os.path.join(output_folder, filename + ".png")
        print(" ==== just before savefig ===== ") 
        print(" ==== filename "+filename) 
        fig.savefig(filename, dpi=plot_settings['dpi'])


################################################################################################
        
if __name__ == '__main__':

    print("*** SCATTER 3D ***")
    set_logger()

    # Load dictionary storing data, configuration and plot settings
    with open(config_filename, "r") as f:
        settings = yaml.safe_load(f)

    # Load coordinate datasets once
    logger.info("Loading coordinate datasets")
    data = settings.pop("data", {})
    x = np.fromfile(data['x'], dtype='float')
    y = np.fromfile(data['y'], dtype='float')
    z = np.fromfile(data['z'], dtype='float')

    options = settings.get("miscellaneous", {})
    if options["verbose"]:
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        zmin, zmax = np.min(z), np.max(z)
        logger.info(f"    Xmin: {xmin:.6f}, Xmax: {xmax:.6f}")
        logger.info(f"    Ymin: {ymin:.6f}, Ymax: {ymax:.6f}")
        logger.info(f"    Zmin: {zmin:.6f}, Zmax: {zmax:.6f}")

    # Loop over all configurations and plot data
    configs = settings.pop("configs", {})
    logger.info(f"Looping over all {len(configs)} configurations")
    camera_settings = settings.pop("camera_settings", {})
    plot_settings = settings.pop("plot_settings", {})

    for c, cfg in enumerate(configs):
        logger.info(f"  - Configuration #{c+1}: Minf = {cfg['Minf']}, AoA = {cfg['AoA']}")
        plot_scatter3D(
            x=x, y=y, z=z,
            config=cfg,
            camera_settings=camera_settings,
            plot_settings=plot_settings,
            **options
        )

###############################################################################       
