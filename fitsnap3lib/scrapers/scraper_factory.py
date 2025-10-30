from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.scrapers.scrape import Scraper
from fitsnap3lib.scrapers.json_scraper import Json
from fitsnap3lib.scrapers.xyz_scraper import XYZ
from fitsnap3lib.scrapers.vasp_scraper import Vasp

# only import LMDB scraper if required modules are available
import importlib.util
if (importlib.util.find_spec("lmdb") is not None and 
    importlib.util.find_spec("ase") is not None):
    from fitsnap3lib.scrapers.fairchem_scraper import FAIRChem
else:
    FAIRChem = None

# only import ADIOS2 scraper if adios2 is available
if importlib.util.find_spec("adios2") is not None:
    from fitsnap3lib.scrapers.adios2_scraper import ADIOS2
else:
    ADIOS2 = None


def scraper(scraper_name, pt, config):
    """Section Factory"""
    instance = search(scraper_name)
    instance.__init__(scraper_name, pt, config)
    return instance


def search(scraper_name):
    instance = None
    for cls in Scraper.__subclasses__():
        if cls.__name__.lower() == scraper_name.lower():
            instance = Scraper.__new__(cls)

    if instance is None:
        raise IndexError("{} was not found in fitsnap scrapers".format(scraper_name))
    else:
        return instance
