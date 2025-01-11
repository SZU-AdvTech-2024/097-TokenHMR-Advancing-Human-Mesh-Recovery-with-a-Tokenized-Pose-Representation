
# Install engine
pip_install_editable "engine"

# Install cv, handling dependencies
pip_install_editable "cv"
pip install -r "cv/requirements/optional.txt"  # Install optional requirements

# Install seg
pip_install_editable "seg"

print_green "Installation done!"

#  使用read命令达到类似bat中的pause命令效果
echo 按任意键继续
read -n 1
echo 继续运行