package final_project

import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.graphx._
import org.jgrapht.alg.matching.blossom.v5.KolmogorovMinimumWeightPerfectMatching
import org.jgrapht.graph.{DefaultWeightedEdge, SimpleWeightedGraph}
import org.apache.spark.storage.StorageLevel
import org.apache.log4j.{Level, Logger}

object BlossomProgram {
  val rootLogger = Logger.getRootLogger()
  rootLogger.setLevel(Level.ERROR)

  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.spark-project").setLevel(Level.WARN)

  def computeMaximumMatching(edges: RDD[(Long, Long)]): RDD[(Long, Long)] = {
    // Convert RDD to a local collection to use with JGraphT
    val edgeList = edges.collect()

    // Create a JGraphT graph
    val graph = new SimpleWeightedGraph[Long, DefaultWeightedEdge](classOf[DefaultWeightedEdge])

    // Add vertices and edges to the graph
    edgeList.foreach { case (src, dst) =>
      if (!graph.containsVertex(src)) graph.addVertex(src)
      if (!graph.containsVertex(dst)) graph.addVertex(dst)
      val edge = graph.addEdge(src, dst)
      graph.setEdgeWeight(edge, 1.0) // Setting weight to 1, as this is an unweighted problem contextually
    }

    // Apply the Blossom algorithm
    val matchingAlgorithm = new KolmogorovMinimumWeightPerfectMatching(graph)
    val matching = matchingAlgorithm.getMatching

    // Convert the matching result back to RDD
    val matchedEdges = matching.getEdges.toArray.map { edge =>
      val source = graph.getEdgeSource(edge.asInstanceOf[DefaultWeightedEdge])
      val target = graph.getEdgeTarget(edge.asInstanceOf[DefaultWeightedEdge])
      (source, target)
    }

    edges.sparkContext.parallelize(matchedEdges)
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("final_project")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder.config(conf).getOrCreate()

    if (args.length != 3) {
      println("Usage: BlossomProgram compute graph_path output_path")
      sys.exit(1)
    }

    val startTimeMillis = System.currentTimeMillis()

    // Read edges from the file and convert them into pairs
    val edges = sc.textFile(args(1)).map { line =>
      val parts = line.split(",")
      (parts(0).toLong, parts(1).toLong)
    }

    val matchedEdges = computeMaximumMatching(edges)

    val endTimeMillis = System.currentTimeMillis()
    val durationSeconds = (endTimeMillis - startTimeMillis) / 1000
    println(s"Blossom algorithm completed in $durationSeconds s.")

    // Save the matched edges
    matchedEdges.map { case (src, dst) => s"$src,$dst" }.coalesce(1).saveAsTextFile(args(2))

    println(s"Matching results saved to ${args(2)}")
  }
}

