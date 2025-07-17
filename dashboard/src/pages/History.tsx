import { useState } from "react";
import type { ChangeAuthor, RepoModification } from "../types";

const AUTHORS: ChangeAuthor[] = [
  {
    name: "Surya Jasper",
    profileUrl: "https://avatars.githubusercontent.com/u/45545431?v=4",
  },
  {
    name: "James Smith",
    profileUrl: "https://avatars.githubusercontent.com/u/45545432?v=4",
  },
  {
    name: "Jane Doe",
    profileUrl: "https://avatars.githubusercontent.com/u/45545433?v=4",
  },
  {
    name: "Lip Gallagher",
    profileUrl: "https://avatars.githubusercontent.com/u/45545434?v=4",
  },
  {
    name: "Jules Vinyard",
    profileUrl: "https://avatars.githubusercontent.com/u/45545435?v=4",
  },
];

export default function History() {
  const [modifications, setModifications] = useState<RepoModification[]>([]);
}
