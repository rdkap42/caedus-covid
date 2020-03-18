# Caedus Covid Frontend
This directory contains the Frontend for Caedus Covid.


## Usage
### Requirements
- Node.js 13.x

### Running the app
To run the Frontend on its own, run:
```
docker-compose run frontend
```

### Other commands
#### Managing dependencies
We use Yarn to manage dependencies. The most common Yarn commands and their NPM equivalents are shown below.

| Yarn command | npm command |
| - | - |
| `yarn add <packages>` | `npm install <packages>` |
| `yarn add --dev <packages>` | `npm install --save-dev <packages>` |
| `yarn remove <packages>` | `npm uninstall <packages>` |
| `yarn` | `npm install` |
| `yarn <command>` | `npm run <command>` |

#### Run the development server
```bash
docker-compose run --rm frontend yarn dev
```

#### Build for production
```bash
docker-compose run --rm frontend yarn build
```

#### Launch the production server
```bash
docker-compose run --rm frontend yarn start
```

#### Lint and fix errors
```bash
docker-compose run --rm frontend yarn lint --fix
```
