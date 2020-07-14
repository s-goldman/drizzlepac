"""Code that evaluates the quality of products generated by the drizzlepac package.

The JSON files generated here can be converted directly into a Pandas DataFrame
using the syntax:

>>> import json
>>> import pandas as pd
>>> with open("<rootname>_astrometry_resids.json") as jfile:
>>>     resids = json.load(jfile)
>>> pdtab = pd.DataFrame(resids)

These DataFrames can then be concatenated using:

>>> allpd = pdtab.concat([pdtab2, pdtab3])

where 'pdtab2' and 'pdtab3' are DataFrames generated from other datasets.  For
more information on how to merge DataFrames, see 

https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html

Visualization of these Pandas DataFrames with Bokeh can follow the example
from:

https://programminghistorian.org/en/lessons/visualizing-with-bokeh


From w3schools.com to go with sample Bokeh code from bottom of 
https://docs.bokeh.org/en/latest/docs/user_guide/bokehjs.html:

<p>Click the button to open a new window called "MsgWindow" with some text.</p>

<button onclick="myFunction()">Try it</button>

<script>
function myFunction() {
  var myWindow = window.open("", "MsgWindow", "width=200,height=100");
  myWindow.document.write("<p>This is 'MsgWindow'. I am 200px wide and 100px tall!</p>");myWindow.document.title="New Window for Plotting";
}
</script>



"""
import json
import os
import sys
from datetime import datetime
import time

from bokeh.layouts import row, column
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, Label
from bokeh.models.tools import HoverTool

from astropy.table import Table, vstack
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import numpy as np

from stwcs.wcsutil import HSTWCS
from stsci.tools.fileutil import countExtn
from stsci.tools import logutil
import tweakwcs

from . import astrometric_utils as amutils
from .. import tweakutils
from . import diagnostic_utils as du
from .pandas_utils import PandasDFReader



MSG_DATEFMT = '%Y%j%H%M%S'
SPLUNK_MSG_FORMAT = '%(asctime)s %(levelname)s src=%(name)s- %(message)s'
log = logutil.create_logger(__name__, level=logutil.logging.NOTSET, stream=sys.stdout,
                            format=SPLUNK_MSG_FORMAT, datefmt=MSG_DATEFMT)
                            
__taskname__ = 'quality_analysis'

__taskname__ = "quality_analysis"

def determine_alignment_residuals(input, files, max_srcs=2000, 
                                  json_timestamp=None,
                                  json_time_since_epoch=None,
                                  log_level=logutil.logging.INFO):
    """Determine the relative alignment between members of an association.

    Parameters
    -----------
    input : string
        Original pipeline input filename.  This filename will be used to
        define the output analysis results filename.

    files : list
        Set of files on which to actually perform comparison.  The original
        pipeline can work on both CTE-corrected and non-CTE-corrected files,
        but this comparison will only be performed on CTE-corrected
        products when available.

    json_timestamp: str, optional
        Universal .json file generation date and time (local timezone) that will be used in the instantiation
        of the HapDiagnostic object. Format: MM/DD/YYYYTHH:MM:SS (Example: 05/04/2020T13:46:35). If not
        specified, default value is logical 'None'

    json_time_since_epoch : float
        Universal .json file generation time that will be used in the instantiation of the HapDiagnostic
        object. Format: Time (in seconds) elapsed since January 1, 1970, 00:00:00 (UTC). If not specified,
        default value is logical 'None'

    log_level : int, optional
        The desired level of verboseness in the log statements displayed on the screen and written to the
        .log file. Default value is 'NOTSET'.

    Returns
    --------
    resids_files : list of string
        Name of JSON files containing all the extracted results from the comparisons
        being performed.
    """
    log.setLevel(log_level)
    
    # Open all files as HDUList objects
    hdus = [fits.open(f) for f in files]
    # Determine sources from each chip
    src_cats = []
    num_srcs = []
    for hdu in hdus:
        numsci = countExtn(hdu)
        nums = 0
        img_cats = {}
        for chip in range(numsci):
            chip += 1
            img_cats[chip] = amutils.extract_point_sources(hdu[("SCI", chip)].data, nbright=max_srcs)
            nums += len(img_cats[chip])

        log.info("Identified {} point-sources from {}".format(nums, hdu.filename()))
        num_srcs.append(nums)
        src_cats.append(img_cats)

    if len(num_srcs) == 0 or (len(num_srcs) > 0 and  max(num_srcs) <= 3):
        log.warning("Not enough sources identified in input images for comparison")
        return None

    # src_cats = [amutils.generate_source_catalog(hdu) for hdu in hdus]
    # Combine WCS from HDULists and source catalogs into tweakwcs-compatible input
    imglist = []
    for i, (f, cat) in enumerate(zip(files, src_cats)):
        imglist += amutils.build_wcscat(f, i, cat)

    # Setup matching algorithm using parameters tuned to well-aligned images
    match = tweakwcs.TPMatch(searchrad=5, separation=4.0,
                             tolerance=1.0, use2dhist=True)
    try:
        # perform relative fitting
        matchlist = tweakwcs.align_wcs(imglist, None, match=match, expand_refcat=False)
        del matchlist
    except Exception:
        try:
            # Try without 2dHist use to see whether we can get any matches at all
            match = tweakwcs.TPMatch(searchrad=5, separation=4.0,
                                     tolerance=1.0, use2dhist=False)
            matchlist = tweakwcs.align_wcs(imglist, None, match=match, expand_refcat=False)
            del matchlist

        except Exception:    
            log.warning("Problem encountered during matching of sources")
            return None
            
    # Check to see whether there were any successful fits...
    align_success = False
    for img in imglist:
        wcsname = fits.getval(img.meta['filename'], 'wcsname', ext=("sci",1))
        img.meta['wcsname'] = wcsname

    for img in imglist:        
        if img.meta['fit_info']['status'] == 'SUCCESS' and '-FIT' in wcsname:
            align_success = True
            break
    resids_files = []
    if align_success:

        # extract results in the style of 'tweakreg'
        resids = extract_residuals(imglist)

        if resids is not None:
            resids_files = generate_output_files(resids, 
                                 json_timestamp=json_timestamp, 
                                 json_time_since_epoch=json_time_since_epoch, 
                                 exclude_fields=['group_id'])

    return resids_files

def generate_output_files(resids_dict, 
                         json_timestamp=None, 
                         json_time_since_epoch=None, 
                         exclude_fields=['group_id'],
                         calling_name='determine_alignment_residuals'):
    """Write out results to JSON files, one per image"""
    resids_files = []
    for image in resids_dict:
        # Remove any extraneous information from output 
        for field in exclude_fields:
            del resids_dict[image]['fit_results'][field]
        # Define name for output JSON file...
        rootname = image.split("_")[0]
        json_filename = "{}_cal_qa_astrometry_resids.json".format(rootname)
        resids_files.append(json_filename)
        
        # Define output diagnostic object
        diagnostic_obj = du.HapDiagnostic()
        src_str = "{}.{}".format(__taskname__, calling_name) 
        diagnostic_obj.instantiate_from_fitsfile(image,
                                               data_source=src_str,
                                               description="X and Y residuals from \
                                                            relative alignment ",
                                               timestamp=json_timestamp,
                                               time_since_epoch=json_time_since_epoch)
        diagnostic_obj.add_data_item(resids_dict[image]['fit_results'], 'fit_results',
                                     item_description="Fit results for relative alignment of input exposures",
                                     descriptions={"aligned_to":"Reference image for relative alignment",
                                                   "rms_x":"RMS in X for fit",
                                                   "rms_y":"RMS in Y for fit",
                                                   "xsh":"X offset from fit",
                                                   "ysh":"Y offset from fit",
                                                   "rot":"Average Rotation from fit",
                                                   "scale":"Average Scale change from fit",
                                                   "rot_fit":"Rotation of each axis from fit",
                                                   "scale_fit":"Scale of each axis from fit",
                                                   "nmatches":"Number of matched sources used in fit",
                                                   "skew":"Skew between axes from fit",
                                                   "wcsname":"WCSNAME for image"},
                                     units={"aligned_to":"unitless",
                                            'rms_x':'pixels',
                                            'rms_y':'pixels',
                                            'xsh':'pixels',
                                            'ysh':'pixels',
                                            'rot':'degrees',
                                            'scale':'unitless',
                                            'rot_fit':'degrees',
                                            'scale_fit':'unitless',
                                            'nmatches':'unitless',
                                            'skew':'unitless',
                                            'wcsname':"unitless"}
                                     )
        diagnostic_obj.add_data_item(resids_dict[image]['sources'], 'residuals',
                                     item_description="Matched source positions from input exposures",
                                     descriptions={"x":"X position from source image on tangent plane",
                                                   "y":"Y position from source image on tangent plane",
                                                   "ref_x":"X position from ref image on tangent plane",
                                                   "ref_y":"Y position from ref image on tangent plane"},
                                     units={'x':'pixels',
                                            'y':'pixels',
                                            'ref_x':'pixels',
                                            'ref_y':'pixels'}
                                     )

        diagnostic_obj.write_json_file(json_filename)
        log.info("Generated relative astrometric residuals for {} as:\n    {}.".format(image, json_filename))

    return resids_files

def extract_residuals(imglist):
    """Convert fit results and catalogs from tweakwcs into list of residuals"""
    group_dict = {}

    ref_ra, ref_dec = [], []
    for chip in imglist:
        group_id = chip.meta['group_id']
        group_name = chip.meta['filename']
        fitinfo = chip.meta['fit_info']
        wcsname = chip.meta['wcsname']

        if fitinfo['status'] == 'REFERENCE':
            align_ref = group_name
            #group_dict[group_name]['aligned_to'] = 'self'
            rra, rdec = chip.det_to_world(chip.meta['catalog']['x'],
                                          chip.meta['catalog']['y'])
            ref_ra = np.concatenate([ref_ra, rra])
            ref_dec = np.concatenate([ref_dec, rdec])
            continue

        if group_id not in group_dict:
            group_dict[group_name] = {}
            group_dict[group_name]['fit_results'] = {'group_id': group_id,
                         'rms_x': None, 'rms_y': None, 'wcsname': wcsname}
            group_dict[group_name]['sources'] = Table(names=['x', 'y', 
                                                             'ref_x', 'ref_y'])
            cum_indx = 0

        # store results in dict
        group_dict[group_name]['fit_results']['aligned_to'] = align_ref

        if 'fitmask' in fitinfo:
            img_mask = fitinfo['fitmask']
            ref_indx = fitinfo['matched_ref_idx'][img_mask]
            img_indx = fitinfo['matched_input_idx'][img_mask]
            # Extract X, Y for sources image being updated
            img_x, img_y, max_indx, chip_mask = get_tangent_positions(chip, img_indx,
                                                           start_indx=cum_indx)
            cum_indx += max_indx
            
            # Extract X, Y for sources from reference image
            ref_x, ref_y = chip.world_to_tanp(ref_ra[ref_indx][chip_mask], ref_dec[ref_indx][chip_mask])
            group_dict[group_name]['fit_results'].update(
                 {'xsh': fitinfo['shift'][0], 'ysh': fitinfo['shift'][1],
                 'rot': fitinfo['<rot>'], 'scale': fitinfo['<scale>'],
                 'rot_fit': fitinfo['rot'], 'scale_fit': fitinfo['scale'],
                 'nmatches': fitinfo['nmatches'], 'skew': fitinfo['skew'],
                 'rms_x': sigma_clipped_stats((img_x - ref_x))[-1],
                 'rms_y': sigma_clipped_stats((img_y - ref_y))[-1]})

            new_vals = Table(data=[img_x, img_y, ref_x, ref_y], 
                                    names=['x', 'y', 'ref_x', 'ref_y'])
            group_dict[group_name]['sources'] = vstack([group_dict[group_name]['sources'], new_vals])
            
        else: 
            group_dict[group_name]['fit_results'].update(
                     {'xsh': None, 'ysh': None,
                     'rot': None, 'scale': None,
                     'rot_fit': None, 'scale_fit': None,
                     'nmatches': -1, 'skew': None,
                     'rms_x': -1, 'rms_y': -1})


    return group_dict

def get_tangent_positions(chip, indices, start_indx=0):
    img_x = []
    img_y = []
    fitinfo = chip.meta['fit_info']
    img_ra = fitinfo['fit_RA']
    img_dec = fitinfo['fit_DEC']

    # Extract X, Y for sources image being updated
    max_indx = len(chip.meta['catalog'])
    chip_indx = np.where(np.logical_and(indices >= start_indx,
                                        indices < max_indx + start_indx))[0]
    # Get X,Y position in tangent plane where fit was done
    chip_x, chip_y = chip.world_to_tanp(img_ra[chip_indx], img_dec[chip_indx])
    img_x.extend(chip_x)
    img_y.extend(chip_y)

    return img_x, img_y, max_indx, chip_indx


# -------------------------------------------------------------------------------
# Compare source list with GAIA ref catalog
def match_to_gaia(imcat, refcat, product, output, searchrad=5.0):
    """Create a catalog with sources matched to GAIA sources
    
    Parameters
    ----------
    imcat : str or obj
        Filename or astropy.Table of source catalog written out as ECSV file
        
    refcat : str
        Filename of GAIA catalog files written out as ECSV file
        
    product : str
        Filename of drizzled product used to derive the source catalog
        
    output : str
        Rootname for matched catalog file to be written as an ECSV file 
    
    """
    if isinstance(imcat, str):
        imtab = Table.read(imcat, format='ascii.ecsv')
        imtab.rename_column('X-Center', 'x')
        imtab.rename_column('Y-Center', 'y')
    else:
        imtab = imcat
        if 'X-Center' in imtab.colnames:
            imtab.rename_column('X-Center', 'x')
            imtab.rename_column('Y-Center', 'y')
            
    
    reftab = Table.read(refcat, format='ascii.ecsv')
    
    # define WCS for matching
    tpwcs = tweakwcs.FITSWCS(HSTWCS(product, ext=1))
    
    # define matching parameters
    tpmatch = tweakwcs.TPMatch(searchrad=searchrad)
    
    # perform match
    ref_indx, im_indx = tpmatch(reftab, imtab, tpwcs)
    print('Found {} matches'.format(len(ref_indx)))
    
    # Obtain tangent plane positions for both image sources and reference sources
    im_x, im_y = tpwcs.det_to_tanp(imtab['x'][im_indx], imtab['y'][im_indx])
    ref_x, ref_y = tpwcs.world_to_tanp(reftab['RA'][ref_indx], reftab['DEC'][ref_indx])
    if 'RA' not in imtab.colnames:
        im_ra, im_dec = tpwcs.det_to_world(imtab['x'][im_indx], imtab['y'][im_indx])
    else:
        im_ra = imtab['RA'][im_indx]
        im_dec = imtab['DEC'][im_indx]
        

    # Compile match table
    match_tab = Table(data=[im_x, im_y, im_ra, im_dec, 
                            ref_x, ref_y, 
                            reftab['RA'][ref_indx], reftab['DEC'][ref_indx]],
                      names=['img_x','img_y', 'img_RA', 'img_DEC', 
                             'ref_x', 'ref_y', 'ref_RA', 'ref_DEC'])
    if not output.endswith('.ecsv'):
        output = '{}.ecsv'.format(output)                             
    match_tab.write(output, format='ascii.ecsv')
    
                       


# -------------------------------------------------------------------------------
# Simple interface for running all the analysis functions defined for this package
def run_all(input, files, log_level=logutil.logging.NOTSET):

    # generate a timestamp values that will be used to make creation time, creation date and epoch values
    # common to each json file
    json_timestamp = datetime.now().strftime("%m/%d/%YT%H:%M:%S")
    json_time_since_epoch = time.time()

    json_files = determine_alignment_residuals(input, files,
                                             json_timestamp=json_timestamp,
                                             json_time_since_epoch=json_time_since_epoch,
                                             log_level=log_level)
    
    print("Generated quality statistics as {}".format(json_files))


# -------------------------------------------------------------------------------
#
#  Code for generating relevant plots from these results
#
# -------------------------------------------------------------------------------
def generate_plots(json_data):
    """Create plots from json file or json data"""
    
    if isinstance(json_data, str):
        # Open json file and read in data
        with open(json_data) as jfile:
            json_data = json.load(jfile)
            
    fig_id = 0
    for fname in json_data:
        data = json_data[fname]

        rootname = fname.split("_")[0]
        coldata = [data['x'], data['y'], data['ref_x'], data['ref_y']]
        # Insure all columns are numpy arrays
        coldata = [np.array(c) for c in coldata]
        title_str = 'Residuals\ for\ {0}\ using\ {1:6d}\ sources'.format(
                    fname.replace('_','\_'),data['nmatches'])
        
        vector_name = '{}_vector_quality.png'.format(rootname)
        resids_name = '{}_resids_quality.png'.format(rootname)
        # Generate plots
        tweakutils.make_vector_plot(None, data=coldata,
                     figure_id=fig_id, title=title_str, vector=True,
                     plotname=vector_name)
        fig_id += 1
        tweakutils.make_vector_plot(None, data=coldata, ylimit=0.5,
                     figure_id=fig_id, title=title_str, vector=False,
                     plotname=resids_name)
        fig_id += 1

# -------------------------------------------------------------------------------
# Generate the Bokeh plot for the pipeline astrometric data.
#
HOVER_COLUMNS = ['gen_info.imgname',
                 'gen_info.instrument',
                 'gen_info.detector',
                 'gen_info.filter',                
                 'header.DATE-OBS',
                 'header.RA_TARG',
                 'header.DEC_TARG',
                 'header.GYROMODE',
                 'header.EXPTIME',
                 'fit_results.aligned_to']
                   
TOOLTIPS_LIST = ['EXPNAME', 'INSTRUMENT', 'DET', 'FILTER', 
                  'DATE', 'RA', 'DEC', 'GYRO', 'EXPTIME',
                  'ALIGNED_TO']
INSTRUMENT_COLUMN = 'full_instrument'

RESULTS_COLUMNS = ['fit_results.rms_x',
                   'fit_results.rms_y',
                   'fit_results.xsh',
                   'fit_results.ysh',
                   'fit_results.rot',
                   'fit_results.scale',
                   'fit_results.rot_fit',
                   'fit_results.scale_fit',
                   'fit_results.nmatches',
                   'fit_results.skew']

RESIDS_COLUMNS = ['residuals.x',
                  'residuals.y',
                  'residuals.ref_x',
                  'residuals.ref_y']
TOOLSEP_START = '{'
TOOLSEP_END = '}'
DETECTOR_LEGEND = {'UVIS': 'magenta', 'IR': 'red', 'WFC': 'blue', 
                    'SBC': 'yellow', 'HRC': 'black'}
    
FIGURE_TOOLS = 'pan,wheel_zoom,box_zoom,zoom_in,zoom_out,box_select,undo,reset,save'
"""
This code comes from:

 https://github.com/holoviz/holoviews/blob/master/holoviews/plotting/bokeh/chart.py#L309

This code shows how they define the arrow heads for the vector plots.  A
'non-invasive' way to use the Bokeh Segment Glyph to create a vector plot would
be to define the arrow heads as 2 additional segments starting at the endpoint
of the vector. The new segments would then be added to the data points to be 
plotted by Segment to appear as an arrow head on each segment. 

        # Get x, y, angle, magnitude and color data
        rads = element.dimension_values(2)
        if self.invert_axes:
            xidx, yidx = (1, 0)
            rads = np.pi/2 - rads
        else:
            xidx, yidx = (0, 1)
        lens = self._get_lengths(element, ranges)/input_scale
        cdim = element.get_dimension(self.color_index)
        cdata, cmapping = self._get_color_data(element, ranges, style,
                                               name='line_color')

        # Compute segments and arrowheads
        xs = element.dimension_values(xidx)
        ys = element.dimension_values(yidx)

        # Compute offset depending on pivot option
        xoffsets = np.cos(rads)*lens/2.
        yoffsets = np.sin(rads)*lens/2.
        if self.pivot == 'mid':
            nxoff, pxoff = xoffsets, xoffsets
            nyoff, pyoff = yoffsets, yoffsets
        elif self.pivot == 'tip':
            nxoff, pxoff = 0, xoffsets*2
            nyoff, pyoff = 0, yoffsets*2
        elif self.pivot == 'tail':
            nxoff, pxoff = xoffsets*2, 0
            nyoff, pyoff = yoffsets*2, 0
        x0s, x1s = (xs + nxoff, xs - pxoff)
        y0s, y1s = (ys + nyoff, ys - pyoff)

        color = None
        if self.arrow_heads:
            arrow_len = (lens/4.)
            xa1s = x0s - np.cos(rads+np.pi/4)*arrow_len
            ya1s = y0s - np.sin(rads+np.pi/4)*arrow_len
            xa2s = x0s - np.cos(rads-np.pi/4)*arrow_len
            ya2s = y0s - np.sin(rads-np.pi/4)*arrow_len
            x0s = np.tile(x0s, 3)
            x1s = np.concatenate([x1s, xa1s, xa2s])
            y0s = np.tile(y0s, 3)
            y1s = np.concatenate([y1s, ya1s, ya2s])
            if cdim and cdim.name in cdata:
                color = np.tile(cdata[cdim.name], 3)
        elif cdim:
            color = cdata.get(cdim.name)

        data = {'x0': x0s, 'x1': x1s, 'y0': y0s, 'y1': y1s}
        mapping = dict(x0='x0', x1='x1', y0='y0', y1='y1')
"""


def build_tooltips(tips):
    """Return list of tuples for tooltips to use in hover tool.
    
    Parameters
    ----------
    tips : list
        List of indices for the HOVER_COLUMNS entries to be used as tooltips 
        to be included in the hover tool.

    """
    tools = [(TOOLTIPS_LIST[i], '@{}{}{}'.format(
                                TOOLSEP_START, 
                                HOVER_COLUMNS[i],
                                TOOLSEP_END)) for i in tips]
    
    return tools
    
def get_pandas_data(pandas_filename):
    """Load the harvested data, stored in a CSV file, into local arrays.

    Parameters
    ==========
    pandas_filename: str
        Name of the CSV file created by the harvester.

    Returns
    =======
    phot_data: Pandas dataframe
        Dataframe which is a subset of the input Pandas dataframe written out as
        a CSV file.  The subset dataframe consists of only the requested columns
        and rows where all of the requested columns did not contain NaNs.

    """
    
    # Instantiate a Pandas Dataframe Reader (lazy instantiation)
    # df_handle = PandasDFReader_CSV("svm_qa_dataframe.csv")
    df_handle = PandasDFReader(pandas_filename, log_level=logutil.logging.NOTSET)

    # In this particular case, the names of the desired columns do not
    # have to be further manipulated, for example, to add dataset specific
    # names.
    # 
    # Get the relevant column data, eliminating all rows which have NaNs
    # in any of the relevant columns.
    if pandas_filename.endswith('.h5'):
        fit_data = df_handle.get_columns_HDF5(HOVER_COLUMNS + RESULTS_COLUMNS)
        resids_data = df_handle.get_columns_HDF5(RESIDS_COLUMNS)
    else:
        fit_data = df_handle.get_columns_CSV(HOVER_COLUMNS + RESULTS_COLUMNS)
        resids_data = df_handle.get_columns_CSV(RESIDS_COLUMNS)

    fitCDS = ColumnDataSource(fit_data)
    residsCDS = ColumnDataSource(resids_data)

    return fitCDS, residsCDS
    
def build_circle_plot(**plot_dict):
    """Create figure object for plotting desired columns as a scatter plot with circles
    
    Parameters
    ----------
    source : Pandas ColumnDataSource 
        Object with all the input data 
    
    x, y : str
        Names of X and Y columns of data from data `source`
    
    x_label, y_label : str
        Labels to use for the X and Y axes (respectively)
        
    title : str
        Title of the plot
    
    tips : list of int
        List of indices for the columns from `source` to use as hints 
        in the HoverTool
        
    color : string, optional
        Single color to use for data points in the plot if `colormap` is not used
        
    colormap : bool, optional
        Specify whether or not to use a pre-defined set of colors for the points
        derived from the `colormap` column in the input data `source`
        
    markersize : int, optional
        Size of each marker in the plot
        
    legend_group : str, optional
        If `colormap` is used, this is the name of the column from the input
        data `source` to use for defining the legend for the colors used.  The
        same colors should be assigned to all the same values of data from the 
        column, for example, all 'ACS/WFC' data points from the `instrument`
        column should have a `colormap` column value of `blue`.
    
    """
    # Interpret required elements
    x = plot_dict['x']
    y = plot_dict['y']
    title = plot_dict['title']
    x_label = plot_dict['x_label']
    y_label = plot_dict['y_label']
    tips = plot_dict['tips']
    source = plot_dict['source']
    
    # check for optional elements
    color = plot_dict.get('color', 'blue')
    click_policy = plot_dict.get('click_policy', 'hide')
    markersize = plot_dict.get('markersize', 10)
    colormap = plot_dict.get('colormap', False)
    legend_group = plot_dict.get('legend_group')
    
    # Define a figure object
    p1 = figure(tools=FIGURE_TOOLS)

    if colormap:
        # Add the glyphs
        # This will use the 'colormap' column from 'source' for the colors of 
        # each point.  This column should have been populated by the calling
        # routine. 
        p1.circle(x=x, y=y, source=source, 
                  fill_alpha=0.5, line_alpha=0.5, 
                  size=markersize, color='colormap',
                  legend_group=legend_group)
    else:
        # Add the glyphs
        p1.circle(x=x, y=y, source=source, 
                  fill_alpha=0.5, line_alpha=0.5, 
                  size=markersize)

    p1.legend.click_policy = click_policy
    
    p1.title.text = title
    p1.xaxis.axis_label = x_label
    p1.yaxis.axis_label = y_label

    hover_p1 = HoverTool()
    tools = build_tooltips(tips)
    hover_p1.tooltips = tools
    p1.add_tools(hover_p1)
    
    return p1
    
def build_vector_plot(**plot_dict):
    """Create figure object for plotting desired columns as a scatter plot with circles
    
    Parameters
    ----------
    source : Pandas ColumnDataSource 
        Object with all the input data 
    
    x, y : str
        Names of X and Y columns of data from data `source`
        
    xr, yr : str
        Names of Reference X/Y columns of data from data `source`
    
    x_label, y_label : str
        Labels to use for the X and Y axes (respectively)
        
    title : str
        Title of the plot
            
    color : string, optional
        Single color to use for data points in the plot if `colormap` is not used
                
    markersize : int, optional
        Size of each marker in the plot
            
    """
    # Interpret required elements
    x = plot_dict['x']
    y = plot_dict['y']
    xr = plot_dict['xr']
    yr = plot_dict['yr']
    title = plot_dict['title']
    x_label = plot_dict['x_label']
    y_label = plot_dict['y_label']
    source = plot_dict['source']
    
    # check for optional elements
    color = plot_dict.get('color', 'blue')
    click_policy = plot_dict.get('click_policy', 'hide')

    # Determine 'optimal' range for axes
    x_arr = np.array(source.data[x])
    y_arr = np.array(source.data[y])
        
    dx = x_arr.max() - min(0, x_arr.min())
    dy = y_arr.max() - min(0, y_arr.min())
    xy_max = int(max(dx,dy))
    xyrange = [0, xy_max]
    xy_seg = xy_max * 0.05
    
    # Define a figure object with square aspect ratio
    p1 = figure(tools=FIGURE_TOOLS, x_range=xyrange, y_range=xyrange)

    # Set length of dx=0.1 to be 5% of the width of the plot
    mag = (xy_seg)/0.1 
    
    delta_x = np.array(source.data['dx'])
    delta_y = np.array(source.data['dy'])
    xr = x_arr + (delta_x * mag)
    yr = y_arr + (delta_y * mag)

    rads = np.arctan2(delta_y, delta_x)
    
    leg_x = [xy_seg]
    leg_xr = [xy_seg * 2]
    leg_y = [xy_seg]
    
    # Add the glyphs
    # This will use the 'colormap' column from 'source' for the colors of 
    # each point.  This column should have been populated by the calling
    # routine. 

    p1.segment(source.data[x], source.data[y], 
               xr.tolist(), yr.tolist(),
              color=color, line_width=2)

    # Add scale for vector length
    p1.segment(leg_x, leg_y, leg_xr, leg_y, color='black', line_width=2)
    scale_text = ['0.1 pixels']
    p1.text([xy_seg], [xy_seg/2.], text=scale_text, 
            text_color='black', text_font_size='10px')

    # add 'arrow heads' to the segments
    p1.triangle(xr, yr, angle=rads + np.pi/5., size=[6]*len(xr), fill_color=color)
     
    p1.title.text = title
    p1.xaxis.axis_label = x_label
    p1.yaxis.axis_label = y_label
    
    return p1
 
   
def generate_summary_plots(fitCDS, output='cal_qa_results.html'):
    """Generate the graphics associated with this particular type of data.

    Parameters
    ==========
    fitCDS : ColumnDataSource object
        Dataframe consisting of the relative alignment astrometric fit results
    
    output : str, optional
        Filename for output file with generated plot
         
    Returns
    -------
    output : str
        Name of HTML file where the plot was saved.

    
    NOTES
    -----
    Example from bokeh.org on how to create a tabbed set of plots:
    
    .. code:: python
    
        from bokeh.io import output_file, show
        from bokeh.models import Panel, Tabs
        from bokeh.plotting import figure

        output_file("slider.html")

        p1 = figure(plot_width=300, plot_height=300)
        p1.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=20, color="navy", alpha=0.5)
        tab1 = Panel(child=p1, title="circle")

        p2 = figure(plot_width=300, plot_height=300)
        p2.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=3, color="navy", alpha=0.5)
        tab2 = Panel(child=p2, title="line")

        tabs = Tabs(tabs=[ tab1, tab2 ])

        show(tabs)

    
    """
    # TODO: include the date from the input data as part of the html filename
    # Set the output file immediately as advised by Bokeh.
    output_file(output)

    num_of_datasets = len(fitCDS.data['index'])
    print('Number of datasets: {}'.format(num_of_datasets))
    
    colormap = [DETECTOR_LEGEND[x] for x in fitCDS.data[HOVER_COLUMNS[2]]]
    fitCDS.data['colormap'] = colormap
    inst_det = ["{}/{}".format(i,d) for (i,d) in zip(fitCDS.data[HOVER_COLUMNS[1]], 
                                         fitCDS.data[HOVER_COLUMNS[2]])]
    fitCDS.data[INSTRUMENT_COLUMN] = inst_det
    
    plot_list = []

    p1 = [build_circle_plot(x=RESULTS_COLUMNS[0], y=RESULTS_COLUMNS[1],
                           source=fitCDS,  
                           title='RMS Values',
                           x_label="RMS_X (pixels)",
                           y_label='RMS_Y (pixels)',
                           tips=[0, 1, 2, 3, 8],
                           colormap=True, legend_group=INSTRUMENT_COLUMN)]
    plot_list += p1
                           
    p2 = [build_circle_plot(x=RESULTS_COLUMNS[2], y=RESULTS_COLUMNS[3],
                           source=fitCDS,
                           title='Offsets',
                           x_label = "SHIFT X (pixels)",
                           y_label = 'SHIFT Y (pixels)',
                           tips=[0, 1, 2, 3, 8],
                           colormap=True, legend_group=INSTRUMENT_COLUMN)]
    plot_list += p2

    p3 = [build_circle_plot(x=RESULTS_COLUMNS[8], y=RESULTS_COLUMNS[4],
                           source=fitCDS,
                           title='Rotation',
                           x_label = "Number of matched sources",
                           y_label = 'Rotation (degrees)',
                           tips=[0, 1, 2, 3, 8],
                           colormap=True, legend_group=INSTRUMENT_COLUMN)]
    plot_list += p3

    p4 = [build_circle_plot(x=RESULTS_COLUMNS[8], y=RESULTS_COLUMNS[5],
                           source=fitCDS,
                           title='Scale',
                           x_label = "Number of matched sources",
                           y_label = 'Scale',
                           tips=[0, 1, 2, 3, 8],
                           colormap=True, legend_group=INSTRUMENT_COLUMN)]
    plot_list += p4
    
    p5 = [build_circle_plot(x=RESULTS_COLUMNS[8], y=RESULTS_COLUMNS[9],
                           source=fitCDS,
                           title='Skew',
                           x_label = "Number of matched sources",
                           y_label = 'Skew (degrees)',
                           tips=[0, 1, 2, 3, 8],
                           colormap=True, legend_group=INSTRUMENT_COLUMN)]
    plot_list += p5


    # Save the generated plots to an HTML file define using 'output_file()'
    save(column(plot_list))
                     
    return output
    
def generate_residual_plots(residsCDS, filename, output_dir=None, output=''):
    
    rootname = '_'.join(filename.split("_")[:-1])
    output = '{}_vectors_{}'.format(rootname, output)
    if output_dir is not None:
        output = os.path.join(output_dir, output)
    output_file(output)
    
    delta_x = np.array(residsCDS.data['x']) - np.array(residsCDS.data['xr'])
    delta_y = np.array(residsCDS.data['y']) - np.array(residsCDS.data['yr'])
    residsCDS.data['dx'] = delta_x
    residsCDS.data['dy'] = delta_y
    npoints = len(delta_x)
    if npoints < 2:
        return None
    
    p1 = build_circle_plot(x='x', y='dx',
                           source=residsCDS,  
                           title='Residuals for {} [{} sources]: X vs DX'.format(filename, npoints),
                           x_label="X (pixels)",
                           y_label='Delta[X] (pixels)',
                           tips=[0, 1, 2, 3, 8])

    p2 = build_circle_plot(x='x', y='dy',
                           source=residsCDS,  
                           title='Residuals for {} [{} sources]: X vs DY'.format(filename, npoints),
                           x_label="X (pixels)",
                           y_label='Delta[Y] (pixels)',
                           tips=[0, 1, 2, 3, 8])
    row1 = row(p1,p2)

    p3 = build_circle_plot(x='y', y='dx',
                           source=residsCDS,  
                           title='Residuals for {} [{} sources]: Y vs DX'.format(filename, npoints),
                           x_label="Y (pixels)",
                           y_label='Delta[X] (pixels)',
                           tips=[0, 1, 2, 3, 8])
 
    p4 = build_circle_plot(x='y', y='dy',
                           source=residsCDS,  
                           title='Residuals for {} [{} sources]: Y vs DY'.format(filename, npoints),
                           x_label="Y (pixels)",
                           y_label='Delta[Y] (pixels)',
                           tips=[0, 1, 2, 3, 8])
    row2 = row(p3,p4)
    

    pv = build_vector_plot(x='x', y='y',
                            xr='xr', yr='yr',
                           source=residsCDS,  
                           title='Vector plot of Image - Reference for {}'.format(filename),
                           x_label="X (pixels)",
                           y_label='Y (pixels)')
    
    print('Saving: {}'.format(output))        
    save(column(row1,row2,pv))

    return output

def build_astrometry_plots(pandas_file, 
                           output_dir=None, 
                           output='cal_qa_results.html'):

    fitCDS, residsCDS = get_pandas_data(pandas_file)

    if output_dir is not None:
        summary_filename = os.path.join(output_dir, output)
    else:
        summary_filename = output
        
    # Generate the astrometric plots
    astrometry_plot_name = generate_summary_plots(fitCDS, output=summary_filename)
    
    resids_plot_names = []
    
    for filename,rootname in zip(fitCDS.data[HOVER_COLUMNS[0]], fitCDS.data['index']):
        i = residsCDS.data['index'].tolist().index(rootname)
        resids_dict = {'x': residsCDS.data[RESIDS_COLUMNS[0]][i],
                       'y': residsCDS.data[RESIDS_COLUMNS[1]][i],
                       'xr': residsCDS.data[RESIDS_COLUMNS[2]][i],
                       'yr': residsCDS.data[RESIDS_COLUMNS[3]][i]}
        cds = ColumnDataSource(resids_dict)

        resids_plot_name = generate_residual_plots(cds, filename, 
                                                   output_dir=output_dir,
                                                   output=output)
        resids_plot_names.append(resids_plot_name)

    resids_plot_names = list(filter(lambda a: a != None, resids_plot_names))
    return resids_plot_names

    