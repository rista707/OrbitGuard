# ns3-sat-sim
echo "Building ns3-sat-sim..."
cd ns3-sat-sim || exit 1

if [ "$1" != "--travis" ]; then
  bash build.sh --debug_all || exit 1

fi
cd .. || exit 1

