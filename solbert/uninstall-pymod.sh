if [ ! -e uninstall.info ]; then
	echo "uninstall.info not found"
	exit
fi

tr '\n' '\0' < uninstall.info | xargs -0 sudo rm -f --
rm uninstall.info
