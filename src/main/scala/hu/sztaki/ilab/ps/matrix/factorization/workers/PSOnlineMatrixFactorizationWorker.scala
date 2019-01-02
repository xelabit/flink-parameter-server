package hu.sztaki.ilab.ps.matrix.factorization.workers

import breeze.linalg.DenseVector
import breeze.numerics.pow
import hu.sztaki.ilab.ps.matrix.factorization.factors.{RangedRandomFactorInitializerDescriptor, SGDUpdater}
import hu.sztaki.ilab.ps.{ParameterServerClient, WorkerLogic}
import hu.sztaki.ilab.ps.matrix.factorization.utils.Rating
import hu.sztaki.ilab.ps.matrix.factorization.utils.Vector._
import hu.sztaki.ilab.ps.matrix.factorization.utils.Utils.{ItemId, UserId}

import scala.collection.mutable
import scala.util.Random

/**
  * Realize the worker logic for online matrix factorization with SGD
  *
  * @param numFactors Number of latent factors
  * @param rangeMin Lower bound of the random number generator
  * @param rangeMax Upper bound of the random number generator
  * @param learningRate Learning rate of SGD
  * @param negativeSampleRate Number of negative samples (Ratings with rate = 0) for each positive rating
  * @param userMemory The last #userMemory item seen by the user will not be generated as negative sample
  */
class PSOnlineMatrixFactorizationWorker(numFactors: Int,
                                        rangeMin: Double,
                                        rangeMax: Double,
                                        learningRate: Double,
                                        userMemory: Int,
                                        negativeSampleRate: Int)  extends WorkerLogic[Rating, Vector, (String, Double)]{



  // initialization method and update method
  val factorInitDesc = RangedRandomFactorInitializerDescriptor(numFactors, rangeMin, rangeMax)
  val factorUpdate = new SGDUpdater(learningRate)

  val userVectors = new mutable.HashMap[UserId, Vector]
  val ratingBuffer = new mutable.HashMap[ItemId, mutable.Queue[Rating]]()
  val itemIds = new mutable.ArrayBuffer[ItemId]
  val seenItemsSet = new mutable.HashMap[UserId, mutable.HashSet[ItemId]]
  val seenItemsQueue = new mutable.HashMap[UserId, mutable.Queue[ItemId]]

  override
  def onPullRecv(paramId: ItemId, paramValue: Vector, ps: ParameterServerClient[Vector, (String, Double)]): Unit = {
    val rating = ratingBuffer synchronized {
      ratingBuffer(paramId).dequeue()
    }

    val user = userVectors.getOrElseUpdate(rating.user, factorInitDesc.open().nextFactor(rating.user))
    val item = paramValue

    rating.label match {
      case "train" => {
        val (userDelta, itemDelta) = factorUpdate.delta(rating.rating, user, item)
        val newItem = vectorSum(item, itemDelta)
        userVectors(rating.user) = vectorSum(user, userDelta)
        val loss = getLoss(userVectors(rating.user), newItem, rating.rating)
        ps.output("train", loss)
        ps.push(paramId, itemDelta)
      }
      case "test" => {
        val loss = getLoss(userVectors(rating.user), item, rating.rating)
        ps.output("test", loss)
      }
    }
  }


  override
  def onRecv(data: Rating, ps: ParameterServerClient[Vector, (String, Double)]): Unit = {

    val seenSet = seenItemsSet.getOrElseUpdate(data.user, new mutable.HashSet)
    val seenQueue = seenItemsQueue.getOrElseUpdate(data.user, new mutable.Queue)

    if (seenQueue.length >= userMemory) {
      seenSet -= seenQueue.dequeue()
    }
    seenSet += data.item
    seenQueue += data.item

    ratingBuffer synchronized {
      for(_  <- 1 to Math.min(itemIds.length - seenSet.size, negativeSampleRate)){
        var randomItemId = itemIds(Random.nextInt(itemIds.size))
        while (seenSet contains randomItemId) {
          randomItemId = itemIds(Random.nextInt(itemIds.size))
        }
        ratingBuffer(randomItemId).enqueue(Rating(data.user, randomItemId, 0.0, data.label, data.timestamp))
        ps.pull(randomItemId)
      }

      ratingBuffer.getOrElseUpdate(
        data.item,
        {
          itemIds += data.item
          mutable.Queue[Rating]()
        }).enqueue(data)
    }

    ps.pull(data.item)
  }

  def getLoss(w: Vector, h: Vector, r: Double): Double = {
    val wFactor = new DenseVector(w)
    val hFactor = new DenseVector(h)
    pow(r - (wFactor dot hFactor), 2)
  }
}
