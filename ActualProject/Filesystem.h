
#ifndef FILESYSTEM_H
#define FILESYSTEM_H


#include "defines.h"


struct FilesystemSpecification {

	std::vector<f32> positions; // iterates over 3 items for x y z values
	std::vector<f32> colors; // the same as positions

	std::vector<f32> radius; // iterates over 1 item, 1 radius per 1 sphere
	std::vector<i32> materials; // 1 material over 1 item

};


const char* const jProject{ "Project" };
const char* const jAuthors{ "Authors" };
const char* const jVersion{ "Version" };
const char* const jName{ "Name" };

const char* const jObjects{ "Objects" };
const char* const jPosition{ "position" };
const char* const jColor{ "color" };
const char* const jMaterial{ "material" };
const char* const jRadius{ "radius" };


void displayFilesystemSpecsItem(u32 index, FilesystemSpecification* pSpecs) {
	std::cout << "\n\nPosition: " << pSpecs->positions[index * 3 + 0] << ' ' << pSpecs->positions[index * 3 + 1] << ' ' << pSpecs->positions[index * 3 + 2]
		      << "\nColor: " << pSpecs->colors[index * 3 + 0] << ' ' << pSpecs->colors[index * 3 + 1] << ' ' << pSpecs->colors[index * 3 + 2]
			  << "\nMaterial: " << (i32)pSpecs->materials[index]
			  << "\nRadius: " << pSpecs->radius[index] << '\n\n';
}


class Filesystem {
public:

	void load(const char* path, FilesystemSpecification* pSpecs) {
		std::ifstream file(path);
		if (!file.is_open()) {
			std::cout << "Cannot open file " << path << " !\n";
			return;
		}

		nlohmann::json jsonFile{ nlohmann::json::parse(file) };
		file.close();

		std::cout << jsonFile[jProject][jAuthors] << '\n';
		std::cout << jsonFile[jProject][jVersion] << '\n';
		std::cout << jsonFile[jProject][jName] << '\n';

		u32 i{ 0 };
		for (nlohmann::json& jsonEntity : jsonFile[jObjects]) {
			pSpecs->positions.push_back(jsonFile[jObjects][i][jPosition]["x"].get<f32>());
			pSpecs->positions.push_back(jsonFile[jObjects][i][jPosition]["y"].get<f32>());
			pSpecs->positions.push_back(jsonFile[jObjects][i][jPosition]["z"].get<f32>());
			
			pSpecs->colors.push_back(jsonFile[jObjects][i][jColor]["x"].get<f32>());
			pSpecs->colors.push_back(jsonFile[jObjects][i][jColor]["y"].get<f32>());
			pSpecs->colors.push_back(jsonFile[jObjects][i][jColor]["z"].get<f32>());

			pSpecs->materials.push_back(jsonFile[jObjects][i][jMaterial].get<i32>());

			pSpecs->radius.push_back(jsonFile[jObjects][i][jRadius].get<f32>());

			displayFilesystemSpecsItem(i, pSpecs);

			i++;
		}

		std::cout << "\n\n";
	}

};


#endif
