package hu.sztaki.ilab.ps.matrix.factorization

import hu.sztaki.ilab.ps.matrix.factorization.utils.Rating
import hu.sztaki.ilab.ps.matrix.factorization.utils.Utils.ItemId
import hu.sztaki.ilab.ps.matrix.factorization.utils.Vector._
import org.apache.flink.api.common.functions.RichFlatMapFunction
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction
import org.apache.flink.streaming.api.scala._
import org.apache.flink.util.Collector


class PSOnlineMatrixFactorizationImplicitTest {

}

object PSOnlineMatrixFactorizationImplicitTest{

  val numFactors = 10
  val rangeMin = -0.1
  val rangeMax = 0.1
  val learningRate = 0.01
  val userMemory = 128
  val negativeSampleRate = 9
  val pullLimit = 1500
  val workerParallelism = 4
  val psParallelism = 4
  val iterationWaitTime = 10000

  def main(args: Array[String]): Unit = {

    val input_file_name = args(0)
    val userVector_output_name = args(1)
    val itemVector_output_name = args(2)

    val env = StreamExecutionEnvironment.getExecutionEnvironment
    val data = env.readTextFile(input_file_name)

    val lastFM = data.flatMap(new RichFlatMapFunction[String, Rating] {

      override def flatMap(value: String, out: Collector[Rating]): Unit = {
        val fieldsArray = value.split(",")
        val r = Rating.fromTuple(fieldsArray(1).toInt, fieldsArray(2).toInt, fieldsArray(4).toInt, fieldsArray(3))
        out.collect(r)
      }
    })

    PSOnlineMatrixFactorization.psOnlineMF(
      lastFM,
      numFactors,
      rangeMin,
      rangeMax,
      learningRate,
      negativeSampleRate,
      userMemory,
      pullLimit,
      workerParallelism,
      psParallelism,
      iterationWaitTime)
        .addSink(new RichSinkFunction[Either[(String, Double), (ItemId, Vector)]] {

          var trainLoss = 0.0
          var testLoss = 0.0

          override def invoke(value: Either[(String, Double), (ItemId, Vector)]): Unit = {
            value match {
              case Left((label, loss)) => label match {
                case "train" => trainLoss += loss
                case "test" => testLoss += loss
              }
            }
          }

          override def close(): Unit = {
            val userVectorFile = new java.io.PrintWriter(new java.io.File(userVector_output_name))
            userVectorFile.write(trainLoss.toString + '\n')
            userVectorFile.close()

            val itemVectorFile = new java.io.PrintWriter(new java.io.File(itemVector_output_name))
            itemVectorFile.write(testLoss.toString + '\n')
            itemVectorFile.close()
          }

        }).setParallelism(1)

    env.execute()
  }
}
