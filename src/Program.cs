using ILGPU.Runtime;
using ILGPU;
using System.Globalization;

namespace Wettbewerb
{
    internal class Program
    {
        static int width;
        static int height;
        static double[] radii = Array.Empty<double>();
        static int[] indices = Array.Empty<int>();
        // Forest to check
        const string forest = "01";
        // Number of runs to make
        const int runs = 1000;
        // Weight by r^rScale
        const double rScale = 0;

        const string baseDir = "../../../..";

        static void ShuffleRadii()
        {
            var r = new Random();
            for (int i = 0; i < radii.Length - 1; i++)
            {
                int j = r.Next(i, radii.Length);

                (radii[i], radii[j]) = (radii[j], radii[i]);

                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        static void Main()
        {
            CultureInfo.CurrentCulture = CultureInfo.CreateSpecificCulture("en-US");
            // Initialize ILGPU.
            Context context = Context.Create(builder => builder.Default().EnableAlgorithms());

            Accelerator accelerator = context.GetPreferredDevice(preferCPU: false)
                                      .CreateAccelerator(context);

            Console.WriteLine(accelerator.Name);

            // load / precompile the kernel
            Action<Index1D, SpecializedValue<int>, SpecializedValue<int>, double, ArrayView<Tree>, ArrayView<Position>, ArrayView<Tree>> findPositionsKernel =
                accelerator.LoadAutoGroupedStreamKernel<Index1D, SpecializedValue<int>, SpecializedValue<int>, double, ArrayView<Tree>, ArrayView<Position>, ArrayView<Tree>>(FindPositions);

            // load / precompile the kernel
            Action<Index1D, ArrayView<Tree>, int, int, ArrayView<double>> getSubScoresKernel =
                accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Tree>, int, int, ArrayView<double>>(SubScores);

            var lines = File.ReadAllLines($"{baseDir}/input_files/forest{forest}.txt");
            var dimensions = lines[1].Split(" ");
            // Init testcase
            width = int.Parse(dimensions[0]) / 1;
            height = int.Parse(dimensions[1]) / 1;
            List<double> radiiTmp = new();
            List<int> indicesTmp = new();
            double minRadius = int.MaxValue;
            for (int i = 2; i < lines.Length; i++)
            {
                double radius = double.Parse(lines[i].Split(" ")[0]);
                if (radius < minRadius)
                {
                    minRadius = radius;
                }
                // forest 2: ^0
                for (int j = 0; j < Math.Pow(radius * (forest == "14" ? 4 : 1), rScale); j++)
                {
                    radiiTmp.Add(radius);
                    indicesTmp.Add(i - 2);
                }
            }
            radii = radiiTmp.ToArray();
            indices = indicesTmp.ToArray();
            ShuffleRadii();

            // Get previous best score
            double best_score = 0;
            if (File.Exists($"{baseDir}/result_files/forest{forest}.txt.out"))
            {
                best_score = Score(File.ReadAllLines($"{baseDir}/result_files/forest{forest}.txt.out").Select(l => l.Split(" "))
                    .Select(tokens => new Tree() { x = double.Parse(tokens[0]), y = double.Parse(tokens[1]), r = double.Parse(tokens[2]), label = int.Parse(tokens[3]) }), width, height);
            }
            Console.WriteLine($"Searching in {forest}");
            Console.WriteLine($"Previous best: {best_score}");

            for (int run = 0; run < runs; run++)
            {
                // Init data
                List<Tree> trees = new()
                {
                    new Tree() { x = radii[0], y = radii[0], r = radii[0], label = indices[0] }
                };
                List<Position> type1s = new()
                {
                    new Position() { c1 = 0, walls=0b00000001 },
                    new Position() { c1 = 0, walls=0b00000010 },
                    new Position() { c1 = 0, walls=0b00000100 },
                    new Position() { c1 = 0, walls=0b00001000 }
                };
                // Special cases
#pragma warning disable CS0162 // Unreachable code detected
                if (forest == "06")
                {
                    trees = new List<Tree>
                    {
                        new Tree() { x = 256, y = 256, r = 256, label = 0 },
                        new Tree() { x = 744, y = 8 * (32 + 5 * Math.Sqrt(15)), r = 256, label = 0 }
                    };
                    type1s = new List<Position>
                    {
                        new Position() { c1 = 0, walls=0b00000001 },
                        new Position() { c1 = 0, walls=0b00000010 },
                        new Position() { c1 = 0, walls=0b00000100 },
                        new Position() { c1 = 0, walls=0b00001000 },
                        new Position() { c1 = 0, c2 = 1 },
                        new Position() { c1 = 1, c2 = 0 }
                    };

                }
                else if (forest == "07")
                {
                    trees = new List<Tree>
                    {
                        new Tree() { x = 256, y = 256, r = 256, label = 12 },
                        new Tree() { x = 744, y = 8 * (32 + 5 * Math.Sqrt(15)), r = 256, label = 12 }
                    };
                    type1s = new List<Position>
                    {
                        new Position() { c1 = 0, walls=0b00000001 },
                        new Position() { c1 = 0, walls=0b00000010 },
                        new Position() { c1 = 0, walls=0b00000100 },
                        new Position() { c1 = 0, walls=0b00001000 },
                        new Position() { c1 = 0, c2 = 1 },
                        new Position() { c1 = 1, c2 = 0 }
                    };
                }
#pragma warning restore CS0162 // Unreachable code detected

                // Search until can't no more
                Console.WriteLine(DateTime.Now);
                while (true)
                {
                    ShuffleRadii();
                    bool foundAny = false;
                    double previousRadius = int.MaxValue;
                    for (int radiusIndex = 0; radiusIndex < radii.Length; radiusIndex++)
                    {
                        // Skip if checking a bigger radius than before
                        if (radii[radiusIndex] >= previousRadius)
                        {
                            continue;
                        }
                        else
                        {
                            previousRadius = radii[radiusIndex];
                        }
                        bool isMinRadius = minRadius == radii[radiusIndex];
                        // Load the data.
                        using MemoryBuffer1D<Tree, Stride1D.Dense> treeBuffer = accelerator.Allocate1D(trees.ToArray());
                        using MemoryBuffer1D<Position, Stride1D.Dense> type1Buffer = accelerator.Allocate1D(type1s.ToArray());
                        using MemoryBuffer1D<Tree, Stride1D.Dense> outputBuffer = accelerator.Allocate1D<Tree>(type1s.Count);

                        // Check all type1s
                        findPositionsKernel(type1s.Count, SpecializedValue.New(width), SpecializedValue.New(height), radii[radiusIndex], treeBuffer.View, type1Buffer.View, outputBuffer.View);
                        accelerator.Synchronize();

                        Tree[] output = outputBuffer.GetAsArray1D();
                        Tree best = new() { x = int.MaxValue, r = -1 };
                        Position[] toAdd = Array.Empty<Position>();
                        List<int> indicesToRemove = new();
                        List<Position> type2s = new();

                        // Find best position
                        for (int i = 0; i < output.Length; i++)
                        {
                            // Collision check
                            if (output[i].collision != -1)
                            {
                                // Collision with other circle
                                if (output[i].collision >= 0)
                                {
                                    Tree c1 = trees[type1s[i].c1];

                                    if (type1s[i].c2 == -1)
                                    {
                                        type2s.Add(new Position() { c1 = type1s[i].c1, c2 = output[i].collision });

                                    } else
                                    {
                                        Tree c2 = trees[type1s[i].c2];
                                        Tree other = trees[output[i].collision];

                                        double r1 = c1.r + other.r;
                                        double r2 = c2.r + other.r;
                                        if (DistSquared(c1, other) > DistSquared(c2, other))
                                        {
                                            type2s.Add(new Position() { c1 = type1s[i].c1, c2 = output[i].collision });
                                        }
                                        else
                                        {
                                            type2s.Add(new Position() { c1 = output[i].collision, c2 = type1s[i].c2 });
                                        }
                                    }
                                } else if (output[i].collision == -2)
                                {
                                    type2s.Add(new Position() { c1 = type1s[i].c1, walls = 0b00000001 });
                                    type2s.Add(new Position() { c1 = type1s[i].c1, walls = 0b00000010 });
                                }

                                // Collision and minimal radius => remove position
                                if (isMinRadius)
                                {
                                    indicesToRemove.Add(i);
                                }
                                continue;
                            }

                            // Found a better position
                            if (output[i].x != -1 && output[i].x < best.x)
                            {
                                best = output[i];

                                // Add new positions
                                Position type1 = type1s[i];
                                if (type1.c1 != -1 && type1.c2 != -1)
                                {
                                    toAdd = new Position[] { new Position() { c1 = trees.Count, c2=type1.c1 },
                                                  new Position() { c1 = type1.c1, c2=trees.Count },
                                                  new Position() { c1 = trees.Count, c2=type1.c2 },
                                                  new Position() { c1 = type1.c2, c2=trees.Count }};
                                }
                                else if ((type1.walls & 0b00001100) != 0)
                                {
                                    toAdd = new Position[] {
                                                  new Position() { c1 = trees.Count, walls = 0b00000100 },
                                                  new Position() { c1 = trees.Count, walls = 0b00001000 },
                                                  new Position() { c1 = type1.c1, c2=trees.Count },
                                                  new Position() { c1 = trees.Count, c2=type1.c1 }};
                                }
                                else if ((type1.walls & 0b00000011) != 0)
                                {
                                    toAdd = new Position[] {
                                                  new Position() { c1 = trees.Count, walls = 0b00000001 },
                                                  new Position() { c1 = trees.Count, walls = 0b00000010 },
                                                  new Position() { c1 = type1.c1, c2=trees.Count },
                                                  new Position() { c1 = trees.Count, c2=type1.c1 }};
                                }
                            }
                        }

                        // Check not directly tangent positions
                        if (type2s.Count > 0)
                        {
                            MemoryBuffer1D<Position, Stride1D.Dense> type2Buffer = accelerator.Allocate1D(type2s.ToArray());
                            MemoryBuffer1D<Tree, Stride1D.Dense> output2Buffer = accelerator.Allocate1D<Tree>(type2s.Count);

                            findPositionsKernel(type2s.Count, SpecializedValue.New(width), SpecializedValue.New(height), radii[radiusIndex], treeBuffer.View, type2Buffer.View, output2Buffer.View);
                            accelerator.Synchronize();

                            Tree[] output2 = output2Buffer.GetAsArray1D();
                            Tree best2 = new() { x = int.MaxValue, r = -1 };
                            Position[] toAdd2 = Array.Empty<Position>();
                            List<int> indicesToRemove2 = new();

                            // Find best position
                            for (int i = 0; i < output2.Length; i++)
                            {
                                if (output2[i].collision != -1)
                                {
                                    if (isMinRadius)
                                    {
                                        indicesToRemove2.Add(i);
                                    }
                                    continue;
                                }

                                // Found better poition
                                if (output2[i].x != -1 && output2[i].x < best2.x)
                                {
                                    best2 = output2[i];

                                    Position type2 = type2s[i];
                                    // Corner between 2 circles
                                    if (type2.c1 != -1 && type2.c2 != -1)
                                    {
                                        toAdd2 = new Position[] { new Position() { c1 = trees.Count, c2=type2.c1 },
                                                  new Position() { c1 = type2.c1, c2=trees.Count },
                                                  new Position() { c1 = trees.Count, c2=type2.c2 },
                                                  new Position() { c1 = type2.c2, c2=trees.Count }};
                                    }
                                    // Bottom wall
                                    else if ((type2.walls & 0b00001100) != 0)
                                    {
                                        toAdd2 = new Position[] { new Position() { c1 = trees.Count, walls = 0b00000100 },
                                                  new Position() { c1 = trees.Count, walls = 0b00001000 },
                                                  new Position() { c1 = type2.c1, c2=trees.Count },
                                                  new Position() { c1 = trees.Count, c2=type2.c1 } };
                                    }
                                    // Left wall
                                    else if ((type2.walls & 0b00000011) != 0)
                                    {
                                        toAdd2 = new Position[] {
                                                  new Position() { c1 = trees.Count, walls = 0b00000001 },
                                                  new Position() { c1 = trees.Count, walls = 0b00000010 },
                                                  new Position() { c1 = type2.c1, c2=trees.Count },
                                                  new Position() { c1 = trees.Count, c2=type2.c1 }};
                                    }
                                }
                            }

                            // Find best beween type1s and type2s
                            if (best.r == -1 || best2.x < best.x)
                            {
                                best = best2;
                                toAdd = toAdd2;
                            }
                        }

                        // Remove no longer positions if the minimal radius was checked
                        if (isMinRadius)
                        {
                            for (int i = indicesToRemove.Count - 1; i >= 0; i--)
                            {
                                type1s.RemoveAt(indicesToRemove[i]);
                            }
                        }

                        if (best.r == -1)
                        {
                            if (isMinRadius)
                            {
                                break;
                            }
                            continue;
                        }

                        best.label = indices[radiusIndex];
                        trees.Add(best);
                        type1s.AddRange(toAdd);
                        foundAny = true;
                        break;
                    }

                    if (!foundAny)
                    {
                        break;
                    }
                }
                Console.WriteLine(DateTime.Now);

                MemoryBuffer1D<Tree, Stride1D.Dense> finalTrees = accelerator.Allocate1D(trees.ToArray());
                MemoryBuffer1D<double, Stride1D.Dense> scoresBuffer = accelerator.Allocate1D<double>(trees.Count);

                getSubScoresKernel(trees.Count, finalTrees.View, width, height, scoresBuffer.View);
                accelerator.Synchronize();
                double[] results = scoresBuffer.GetAsArray1D();
                int bestIndex = FindBestResult(results);

                if (results[bestIndex] > best_score)
                {
                    Console.WriteLine($"Found better result with {results[bestIndex]} > {best_score} ===========================================================================");
                    File.WriteAllLines($"{baseDir}/result_files/forest{forest}.txt.out", trees.Where((t, i) => i <= bestIndex).Select(t => $"{t.x} {t.y} {t.r} {t.label}"));
                    best_score = results[bestIndex];
                }
                else
                {
                    Console.WriteLine($"Didn't find better score {results[bestIndex]} < {best_score}");
                }
            }
            Console.WriteLine($"Best result: B={best_score}");


            accelerator.Dispose();
            context.Dispose();
        }

        static int FindBestResult(double[] scores)
        {
            double currentBestScore = 0;
            int bestIndex = 0;
            for (int i = 1; i < scores.Length; i++)
            {
                if (scores[i] > currentBestScore)
                {
                    currentBestScore = scores[i];
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        static void FindPositions(Index1D i, SpecializedValue<int> width, SpecializedValue<int> height, double radius, ArrayView<Tree> currentTrees, ArrayView<Position> possiblePositions, ArrayView<Tree> output)
        {
            Position posToCheck = possiblePositions[i];
            double x = -1;
            double y = -1;
            if (posToCheck.c1 != -1 && posToCheck.c2 != -1)
            {
                // Check between 2 circles
                Tree c1 = currentTrees[posToCheck.c1];
                Tree c2 = currentTrees[posToCheck.c2];
                double a = DistSquared(c1, c2);
                double b = c2.r + radius;
                double c = c1.r + radius;


                double alpha = Math.Acos((a + b * b - c * c) / (2 * Math.Sqrt(a) * b));
                double beta = Math.Atan2(c2.y - c1.y, c2.x - c1.x);

                x = c2.x - (c2.r + radius) * Math.Cos(alpha - beta);
                y = c2.y + (c2.r + radius) * Math.Sin(alpha - beta);
            }
            // Bottom wall
            else if ((posToCheck.walls & 0b00001100) != 0)
            {
                int sign;
                Tree other = currentTrees[posToCheck.c1];
                if ((posToCheck.walls & 0b00000100) != 0)
                {
                    sign = 1;
                }
                else
                {
                    sign = -1;
                }
                double sr = other.r + radius;
                double dr = other.y - radius;

                x = other.x + sign * Math.Sqrt(sr * sr - dr * dr);
                y = radius;
            }
            else if ((posToCheck.walls & 0b00000011) != 0)
            {
                int sign;
                Tree other = currentTrees[posToCheck.c1];
                if ((posToCheck.walls & 0b00000001) != 0)
                {
                    sign = 1;
                }
                else
                {
                    sign = -1;
                }

                double sr = other.r + radius;
                double dr = other.x - radius;

                x = radius;
                y = other.y + sign * Math.Sqrt(sr * sr - dr * dr);

            }

            // Check if out of bounds
            // Hit Left wall
            if (x - radius < 0)
            {
                output[i] = new Tree() { collision = -2 };
                return;
            }
            // Hit bottom wall
            if (y - radius < 0)
            {
                output[i] = new Tree() { collision = -3 };
                return;
            }
            // Top wall
            if (y + radius > height)
            {
                output[i] = new Tree() { collision = -4 };
                return;
            }
            // Right wall
            if (x + radius > width)
            {
                output[i] = new Tree() { collision = -5 };
                return;
            }
            // Check for collision
            int collision = -1;
            for (int otherTreeIndex = 0; otherTreeIndex < currentTrees.Length; otherTreeIndex++)
            {
                double dx = currentTrees[otherTreeIndex].x - x;
                double dy = currentTrees[otherTreeIndex].y - y;
                double r = currentTrees[otherTreeIndex].r + radius;
                if (dx * dx + dy * dy < r * r - 0.0000000001)
                {
                    collision = otherTreeIndex;
                    break;
                }
            }
            if (collision == -1)
            {
                output[i] = new Tree() { x = x, y = y, r = radius };
            }
            else
            {
                output[i] = new Tree() { collision = collision };
            }
        }

        static double Score(IEnumerable<Tree> trees, int width, int height)
        {
            double totalArea = width * height;
            double actualArea = 0;

            Dictionary<int, int> counts = new Dictionary<int, int>();

            foreach (var tree in trees)
            {
                actualArea += Math.PI * tree.r * tree.r;
                if (counts.ContainsKey(tree.label))
                {
                    counts[tree.label] = counts[tree.label] + 1;
                }
                else
                {
                    counts[tree.label] = 1;
                }
            }
            double A = (actualArea / totalArea);

            double D = 0;
            foreach (var item in counts)
            {
                double frac = (double)item.Value / trees.Count();
                D += frac * frac;
            }
            D = 1 - D;
            return A * D;
        }

        static void SubScores(Index1D i, ArrayView<Tree> trees, int width, int height, ArrayView<double> output)
        {
            double totalArea = width * height;
            double actualArea = 0;

            int[] counts = new int[1000];

            for (int j = 0; j <= i; j++)
            {
                Tree tree = trees[j];
                actualArea += Math.PI * tree.r * tree.r;
                counts[tree.label] = counts[tree.label] + 1;

            }
            double A = (actualArea / totalArea);

            double D = 0;
            foreach (var item in counts)
            {
                double frac = (double)item / (i + 1);
                D += frac * frac;
            }
            D = 1 - D;
            output[i] = A * D;
        }

        public static double DistSquared(Tree t1, Tree t2)
        {
            double dx = t1.x - t2.x;
            double dy = t1.y - t2.y;
            return dx * dx + dy * dy;
        }
    }

    public struct Tree
    {
        public double x = -1;
        public double y = -1;
        public double r = -1;
        public int label = -1;
        public int collision = -1;

        public Tree()
        {
        }
    }

    public struct Position
    {
        public int c1 = -1;
        public int c2 = -1;
        public byte walls = 0b00000000;

        public Position()
        {
        }
    }
}