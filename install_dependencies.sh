# General
sudo apt-get update || exit 1
pip install numpy astropy ephem networkx sgp4 geopy matplotlib statsmodels || exit 1
sudo apt-get install libproj-dev proj-data proj-bin libgeos-dev || exit 1
pip install git+https://github.com/snkas/exputilpy.git@v1.6 || exit 1
pip install cartopy || exit 1
echo "Installing dependencies for ns3-sat-sim..."
sudo apt-get -y install openmpi-bin openmpi-common openmpi-doc libopenmpi-dev lcov gnuplot || exit 1
pip install numpy statsmodels || exit 1
pip install git+https://github.com/snkas/exputilpy.git@v1.6 || exit 1
git submodule update --init --recursive || exit 1

