import os
import shutil


def get_files(path, topK=None):
    file_li = []
    for root, dirs, _files in os.walk(path):
        if dirs:
            for _dir in dirs:
                num = 0
                for _, _, files in os.walk(os.path.join(root, _dir)):
                    for file in files:
                        file_li.append(os.path.join(root, _dir, file))
                        if topK:
                            num += 1
                            if num == topK:
                                break
                    break
        # else:
        #     for f in _files:
        #         num = 0
        #         file_li.append(os.path.join(root, f))
        #         if topK:
        #             num += 1
        #             if num == topK:
        #                 break

    return file_li


def recreate_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)


# def copy_files(dest_dir, dest_path, path_list, src_dir):
#     for path in path_list:
#         src_path = src_dir + path
#         dest_path = dest_dir + dest_path
#         shutil.copyfile(src_path, dest_path)


if __name__ == "__main__":
    print(len(get_files("./corpus/test", 2)))
